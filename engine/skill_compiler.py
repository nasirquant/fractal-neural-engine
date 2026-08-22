"""Self-improving skill compiler for FNSE.

Analyzes simulation failure trajectories, compiles successful workflows
into isolated Python script files (skills), and loads them dynamically.
"""

from __future__ import annotations

import ast
import importlib.util
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .state import SkillManifest, AgentState, SimulationTickPacket
from config import settings


@dataclass
class CompilationResult:
    """Result of a skill compilation attempt."""
    success: bool
    skill_id: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureTrajectory:
    """Captures a failure trajectory for analysis."""
    epoch_id: str
    tick_number: int
    agent_id: str
    agent_role: str
    error_type: str
    error_message: str
    stack_trace: str
    agent_state_snapshot: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SkillValidator:
    """Validates compiled skills for safety and correctness."""
    
    def __init__(self):
        self.allowed_imports = set(settings.allowed_imports)
        self.blocked_keywords = set(settings.blocked_keywords)
    
    def validate_source(self, source_code: str) -> Tuple[bool, List[str]]:
        """Validate source code for safety violations."""
        errors = []
        
        # Check for blocked keywords using regex with word boundaries
        import re
        for keyword in self.blocked_keywords:
            # Use word boundaries to avoid matching substrings like "exec" in "execute"
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, source_code):
                errors.append(f"Blocked keyword detected: {keyword}")
        
        # Parse AST for deeper analysis
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]
        
        # Walk AST for imports and dangerous constructs
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        full_name = f"{module}.{alias.name}" if module else alias.name
                    else:
                        full_name = alias.name
                    
                    # Check if import is allowed
                    base_module = full_name.split(".")[0]
                    if base_module not in self.allowed_imports:
                        errors.append(f"Disallowed import: {full_name}")
            
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "__import__"):
                        errors.append(f"Dangerous function call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("eval", "exec", "compile", "__import__"):
                        errors.append(f"Dangerous method call: {node.func.attr}")
        
        return len(errors) == 0, errors
    
    def validate_signature(self, source_code: str, expected_signature: str) -> Tuple[bool, List[str]]:
        """Validate that the skill has the expected entry point signature."""
        errors = []
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return False, ["Cannot parse source for signature validation"]
        
        # Find the entry point function
        entry_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                entry_func = node
                break
        
        if not entry_func:
            return False, ["No 'execute' function found"]
        
        # Check signature matches expected
        args = [arg.arg for arg in entry_func.args.args]
        if "self" in args:
            args.remove("self")
        
        return True, errors


class SkillCompiler:
    """Compiles successful agent workflows into reusable skills."""
    
    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir or "./skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        self._validator = SkillValidator()
        self._loaded_skills: Dict[str, Callable] = {}
        self._skill_manifests: Dict[str, SkillManifest] = {}
        self._failure_trajectories: List[FailureTrajectory] = []
        self._lock = RLock()
        self._compilation_depth = 0
    def analyze_failures(self, trajectories: List[FailureTrajectory]) -> List[Dict[str, Any]]:
        """Analyze failure trajectories to identify patterns for skill extraction."""
        patterns = []
        
        # Group by error type
        by_error: Dict[str, List[FailureTrajectory]] = {}
        for traj in trajectories:
            if traj.error_type not in by_error:
                by_error[traj.error_type] = []
            by_error[traj.error_type].append(traj)
        
        # Identify common patterns
        for error_type, trajs in by_error.items():
            if len(trajs) >= 3:  # At least 3 occurrences
                # Extract common context
                common_keys = set(trajs[0].context.keys())
                for traj in trajs[1:]:
                    common_keys &= set(traj.context.keys())
                
                pattern = {
                    "error_type": error_type,
                    "frequency": len(trajs),
                    "common_context_keys": list(common_keys),
                    "affected_roles": list(set(t.agent_role for t in trajs)),
                    "sample_trajectory": trajs[0],
                }
                patterns.append(pattern)
        
        return patterns

    def extract_workflow(self, epoch_id: str, tick_range: Tuple[int, int], 
                        agent_id: str) -> Optional[str]:
        """Extract a successful workflow from simulation history."""
        return f'''
def execute(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracted workflow from epoch {epoch_id}, ticks {tick_range[0]}-{tick_range[1]},
    agent {agent_id}
    """
    # Workflow steps would be reconstructed here
    result = {{"status": "success", "data": input_data}}
    return result
'''
    
    def compile_skill(self, 
                     name: str,
                     description: str,
                     source_code: str,
                     author_agent_id: str,
                     compilation_epoch: str,
                     compilation_tick: int,
                     parent_skill_ids: Optional[List[str]] = None,
                     test_cases: Optional[List[Dict[str, Any]]] = None) -> CompilationResult:
        """Compile a skill from source code."""
        with self._lock:
            if self._compilation_depth >= settings.max_recursion_depth:
                return CompilationResult(
                    success=False,
                    error=f"Max compilation depth ({settings.max_recursion_depth}) exceeded"
                )
            
            self._compilation_depth += 1
            try:
                return self._compile_skill_internal(
                    name, description, source_code, author_agent_id,
                    compilation_epoch, compilation_tick, parent_skill_ids, test_cases
                )
            finally:
                self._compilation_depth -= 1
    
    def _compile_skill_internal(self,
                               name: str,
                               description: str,
                               source_code: str,
                               author_agent_id: str,
                               compilation_epoch: str,
                               compilation_tick: int,
                               parent_skill_ids: Optional[List[str]] = None,
                               test_cases: Optional[List[Dict[str, Any]]] = None) -> CompilationResult:
        """Internal compilation logic."""
        # Validate source
        valid, errors = self._validator.validate_source(source_code)
        if not valid:
            return CompilationResult(success=False, error="; ".join(errors))
        
        # Validate signature
        valid, sig_errors = self._validator.validate_signature(source_code, "execute")
        if not valid:
            return CompilationResult(success=False, error="; ".join(sig_errors))
        
        # Generate skill ID
        skill_hash = hashlib.sha256(source_code.encode()).hexdigest()[:12]
        skill_id = f"skill_{skill_hash}"
        
        # Create skill file
        skill_file = self.skills_dir / f"{skill_id}.py"
        
        # Add safety wrapper
        wrapped_code = self._wrap_skill(source_code, skill_id)
        
        try:
            skill_file.write_text(wrapped_code)
        except Exception as e:
            return CompilationResult(success=False, error=f"Failed to write skill file: {e}")
        
        # Load and test the skill
        load_result = self._load_skill(skill_id, skill_file)
        if not load_result.success:
            skill_file.unlink(missing_ok=True)
            return load_result
        
        # Run tests
        test_results = {}
        passed = 0
        failed = 0
        
        if test_cases:
            for i, test_case in enumerate(test_cases):
                try:
                    result = self._loaded_skills[skill_id](**test_case.get("input", {}))
                    expected = test_case.get("expected")
                    if expected is None or result == expected:
                        passed += 1
                        test_results[f"test_{i}"] = {"passed": True, "output": result}
                    else:
                        failed += 1
                        test_results[f"test_{i}"] = {
                            "passed": False, 
                            "output": result, 
                            "expected": expected
                        }
                except Exception as e:
                    failed += 1
                    test_results[f"test_{i}"] = {"passed": False, "error": str(e)}
        
        # Create manifest
        manifest = SkillManifest(
            skill_id=skill_id,
            name=name,
            description=description,
            source_code=source_code,
            entry_point="execute",
            signature="(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]",
            author_agent_id=author_agent_id,
            parent_skill_ids=parent_skill_ids or [],
            compilation_tick=compilation_tick,
            compilation_epoch=compilation_epoch,
            test_cases=test_cases or [],
            passed_tests=passed,
            failed_tests=failed,
            allowed_imports=list(self._validator.allowed_imports),
        )
        
        self._skill_manifests[skill_id] = manifest
        
        # Save manifest
        manifest_file = self.skills_dir / f"{skill_id}.manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        
        return CompilationResult(
            success=True,
            skill_id=skill_id,
            test_results=test_results
        )
        valid, sig_errors = self._validator.validate_signature(source_code, "execute")
        if not valid:
            return CompilationResult(success=False, error="; ".join(sig_errors))
        
        # Generate skill ID
        skill_hash = hashlib.sha256(source_code.encode()).hexdigest()[:12]
        skill_id = f"skill_{skill_hash}"
        
        # Create skill file
        skill_file = self.skills_dir / f"{skill_id}.py"
        
        # Add safety wrapper
        wrapped_code = self._wrap_skill(source_code, skill_id)
        
        try:
            skill_file.write_text(wrapped_code)
        except Exception as e:
            return CompilationResult(success=False, error=f"Failed to write skill file: {e}")
        
        # Load and test the skill
        load_result = self._load_skill(skill_id, skill_file)
        if not load_result.success:
            skill_file.unlink(missing_ok=True)
            return load_result
        
        # Run tests
        test_results = {}
        passed = 0
        failed = 0
        
        if test_cases:
            for i, test_case in enumerate(test_cases):
                try:
                    result = self._loaded_skills[skill_id](**test_case.get("input", {}))
                    expected = test_case.get("expected")
                    if expected is None or result == expected:
                        passed += 1
                        test_results[f"test_{i}"] = {"passed": True, "output": result}
                    else:
                        failed += 1
                        test_results[f"test_{i}"] = {
                            "passed": False, 
                            "output": result, 
                            "expected": expected
                        }
                except Exception as e:
                    failed += 1
                    test_results[f"test_{i}"] = {"passed": False, "error": str(e)}
        
        # Create manifest
        manifest = SkillManifest(
            skill_id=skill_id,
            name=name,
            description=description,
            source_code=source_code,
            entry_point="execute",
            signature="(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]",
            author_agent_id=author_agent_id,
            parent_skill_ids=parent_skill_ids or [],
            compilation_tick=compilation_tick,
            compilation_epoch=compilation_epoch,
            test_cases=test_cases or [],
            passed_tests=passed,
            failed_tests=failed,
            allowed_imports=list(self._validator.allowed_imports),
        )
        
        self._skill_manifests[skill_id] = manifest
        
        # Save manifest
        manifest_file = self.skills_dir / f"{skill_id}.manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        
        return CompilationResult(
            success=True,
            skill_id=skill_id,
            test_results=test_results
        )
    
    def _wrap_skill(self, source_code: str, skill_id: str) -> str:
        """Wrap skill code with safety measures."""
        # DEBUG
        print(f"DEBUG _wrap_skill source_code len: {len(source_code)}")
        print(f"DEBUG _wrap_skill source_code has re.sub: {'re.sub' in source_code}")
        if 're.sub' in source_code:
            idx = source_code.find('re.sub')
            print(f"DEBUG _wrap_skill source_code snippet: {repr(source_code[idx:idx+60])}")
        wrapper = '''
"""
Skill: {skill_id}
Auto-generated by FNSE Skill Compiler
"""

import json
import sys
import traceback
from typing import Any, Dict

# Original skill code
{source_code}

# Safe entry point
def safe_execute(*args, **kwargs) -> Dict[str, Any]:
    """Safe wrapper for skill execution."""
    try:
        result = execute(*args, **kwargs)
        return {{"success": True, "result": result, "error": None}}
    except Exception as e:
        return {{
            "success": False, 
            "result": None, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }}

if __name__ == "__main__":
    # Allow CLI testing
    import sys
    if len(sys.argv) > 1:
        input_data = json.loads(sys.argv[1])
        result = safe_execute(**input_data)
        print(json.dumps(result))
'''.format(skill_id=skill_id, source_code=source_code)
        if 're.sub' in wrapper:
            idx = wrapper.find('re.sub')
            print(f"DEBUG _wrap_skill result snippet: {repr(wrapper[idx:idx+60])}")
        return wrapper
    
    def _load_skill(self, skill_id: str, skill_file: Path) -> CompilationResult:
        """Load a skill module dynamically."""
        try:
            spec = importlib.util.spec_from_file_location(skill_id, skill_file)
            if spec is None or spec.loader is None:
                return CompilationResult(success=False, error="Failed to create module spec")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[skill_id] = module
            spec.loader.exec_module(module)
            
            # Get the safe_execute function
            if not hasattr(module, "safe_execute"):
                return CompilationResult(success=False, error="No safe_execute function found")
            
            self._loaded_skills[skill_id] = module.safe_execute
            return CompilationResult(success=True, skill_id=skill_id)
            
        except Exception as e:
            return CompilationResult(success=False, error=f"Failed to load skill: {e}")
    
    def load_skill(self, skill_id: str) -> Optional[Callable]:
        """Load a previously compiled skill."""
        with self._lock:
            if skill_id in self._loaded_skills:
                return self._loaded_skills[skill_id]
            
            # Try to load from disk
            skill_file = self.skills_dir / f"{skill_id}.py"
            if not skill_file.exists():
                return None
            
            result = self._load_skill(skill_id, skill_file)
            if result.success:
                return self._loaded_skills[skill_id]
            return None
    
    def get_manifest(self, skill_id: str) -> Optional[SkillManifest]:
        """Get skill manifest."""
        with self._lock:
            if skill_id in self._skill_manifests:
                return self._skill_manifests[skill_id]
            
            # Try to load from disk
            manifest_file = self.skills_dir / f"{skill_id}.manifest.json"
            if manifest_file.exists():
                try:
                    manifest = SkillManifest.model_validate_json(manifest_file.read_text())
                    self._skill_manifests[skill_id] = manifest
                    return manifest
                except Exception:
                    return None
            return None
    
    def list_skills(self) -> List[SkillManifest]:
        """List all available skills."""
        with self._lock:
            skills = list(self._skill_manifests.values())
            
            # Also check disk for any not in memory
            for manifest_file in self.skills_dir.glob("*.manifest.json"):
                skill_id = manifest_file.stem.replace(".manifest", "")
                if skill_id not in self._skill_manifests:
                    try:
                        manifest = SkillManifest.model_validate_json(manifest_file.read_text())
                        self._skill_manifests[skill_id] = manifest
                        skills.append(manifest)
                    except Exception:
                        pass
            
            return skills
    
    def record_failure(self, trajectory: FailureTrajectory) -> None:
        """Record a failure trajectory for later analysis."""
        with self._lock:
            self._failure_trajectories.append(trajectory)
            
            # Keep only recent trajectories
            max_trajectories = 1000
            if len(self._failure_trajectories) > max_trajectories:
                self._failure_trajectories = self._failure_trajectories[-max_trajectories:]
    
    def auto_compile_from_failures(self, epoch_id: str) -> List[CompilationResult]:
        """Automatically compile skills from recorded failure patterns."""
        results = []
        
        with self._lock:
            trajectories = self._failure_trajectories.copy()
        
        patterns = self.analyze_failures(trajectories)
        
        for pattern in patterns:
            # For each pattern, attempt to generate a remediation skill
            skill_name = f"remediate_{pattern['error_type']}"
            skill_desc = f"Auto-generated remediation for {pattern['error_type']} (seen {pattern['frequency']} times)"
            
            # Generate remediation code template
            source = f'''
def execute(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-generated remediation for {pattern['error_type']}.
    Context keys: {pattern['common_context_keys']}
    """
    # Remediation logic would go here
    return {{"status": "remediated", "error_type": "{pattern['error_type']}", "input": input_data}}
'''
            
            result = self.compile_skill(
                name=skill_name,
                description=skill_desc,
                source_code=source,
                author_agent_id="skill_compiler",
                compilation_epoch=epoch_id,
                compilation_tick=0,
            )
            results.append(result)
        
        return results


class SkillRegistry:
    """Global registry for skill management across simulations."""
    
    def __init__(self):
        self._compilers: Dict[str, SkillCompiler] = {}
        self._lock = RLock()
    
    def get_compiler(self, epoch_id: str, skills_dir: Optional[str] = None) -> SkillCompiler:
        """Get or create a skill compiler for an epoch."""
        with self._lock:
            if epoch_id not in self._compilers:
                self._compilers[epoch_id] = SkillCompiler(skills_dir)
            return self._compilers[epoch_id]
    
    def remove_compiler(self, epoch_id: str) -> bool:
        """Remove a compiler and cleanup."""
        with self._lock:
            if epoch_id in self._compilers:
                del self._compilers[epoch_id]
                return True
            return False


# Global registry
skill_registry = SkillRegistry()
