"""
Rekarisk QRA — Event Tree Analysis.

Event tree modelling for quantitative risk assessment. An event tree
starts from an initiating event and branches through a sequence of
safety system successes/failures, yielding terminal outcome scenarios
with associated probabilities.

References:
  - CCPS Guidelines for Chemical Process Quantitative Risk Analysis
  - TNO Purple Book CPR 18E
  - NUREG/CR-2300 PRA Procedures Guide
  - Lees' Loss Prevention in the Process Industries
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, Union

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class ConsequenceType(str, Enum):
    """Hazard consequence outcome types."""
    DISPERSION = "dispersion"        # Toxic/flammable dispersion (no ignition)
    POOL_FIRE = "pool_fire"         # Liquid pool fire
    JET_FIRE = "jet_fire"           # Pressurised jet fire
    FLASH_FIRE = "flash_fire"       # Delayed ignition of vapour cloud
    EXPLOSION = "explosion"         # Vapour cloud explosion (VCE)
    BLEVE = "bleve"                 # Boiling Liquid Expanding Vapour Explosion
    TOXIC = "toxic"                 # Toxic release (no fire/explosion)
    SAFE_DISPERSAL = "safe_dispersal"  # Safe dispersal, no harm


class BranchType(str, Enum):
    """Type of branching node in event tree."""
    YES_NO = "yes_no"               # Success/failure binary split
    MULTI = "multi"                 # Multiple discrete outcomes
    CONTINUOUS = "continuous"       # Continuous probability distribution


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """Terminal outcome scenario from an event tree.

    Attributes
    ----------
    name : str
        Human-readable scenario identifier.
    description : str
        Narrative description of the scenario sequence.
    probability : float
        Scenario probability = initiating frequency × path probability.
    consequence_type : ConsequenceType
        Type of hazardous consequence.
    consequence_params : dict
        Parameters for the consequence model (release rate, duration, etc.).
    path : list[str]
        Ordered list of branching decisions from root to this outcome.
    category : str
        Risk category classification (optional).
    """
    name: str
    description: str = ""
    probability: float = 0.0
    consequence_type: ConsequenceType = ConsequenceType.DISPERSION
    consequence_params: dict[str, Any] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "probability": self.probability,
            "consequence_type": self.consequence_type.value,
            "consequence_params": self.consequence_params,
            "path": self.path,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            probability=data.get("probability", 0.0),
            consequence_type=ConsequenceType(data.get("consequence_type", "dispersion")),
            consequence_params=data.get("consequence_params", {}),
            path=data.get("path", []),
            category=data.get("category", ""),
        )


@dataclass
class EventTreeNode:
    """A node in the event tree.

    Can be a branching node (with yes/no children) or a terminal outcome.

    Attributes
    ----------
    name : str
        Node name (e.g., "Immediate Ignition?", "Delayed Ignition?").
    description : str
        Description of this branch point.
    branch_type : BranchType
        Type of branching at this node.
    probability_yes : float
        Conditional probability of the "yes"/success branch.
    probability_no : float
        Conditional probability of the "no"/failure branch.
        Default computed as 1 - probability_yes for yes/no branches.
    outcome_name : str
        If this is a terminal node, the scenario name.
    consequence_type : ConsequenceType
        Consequence type for terminal nodes.
    consequence_params : dict
        Additional consequence model parameters.
    children : list[EventTreeNode]
        Child nodes (branches).
    parent : EventTreeNode or None
        Parent node reference.
    is_terminal : bool
        True if this node is a terminal outcome.
    """
    name: str
    description: str = ""
    branch_type: BranchType = BranchType.YES_NO
    probability_yes: float = 0.5
    probability_no: Optional[float] = None   # Auto-computed if None
    outcome_name: str = ""
    consequence_type: ConsequenceType = ConsequenceType.DISPERSION
    consequence_params: dict[str, Any] = field(default_factory=dict)
    children: list["EventTreeNode"] = field(default_factory=list)
    parent: Optional["EventTreeNode"] = None
    is_terminal: bool = False

    @property
    def prob_no(self) -> float:
        """Get probability of 'no' branch (computed if not set)."""
        if self.probability_no is not None:
            return self.probability_no
        return max(0.0, min(1.0, 1.0 - self.probability_yes))

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (terminal)."""
        return len(self.children) == 0 or self.is_terminal

    def add_child(self, child: "EventTreeNode") -> "EventTreeNode":
        """Add a child node and set parent reference."""
        child.parent = self
        self.children.append(child)
        return child

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "branch_type": self.branch_type.value,
            "probability_yes": self.probability_yes,
            "probability_no": self.probability_no,
            "outcome_name": self.outcome_name,
            "consequence_type": self.consequence_type.value,
            "consequence_params": self.consequence_params,
            "is_terminal": self.is_terminal,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventTreeNode":
        """Deserialize node from dictionary."""
        node = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            branch_type=BranchType(data.get("branch_type", "yes_no")),
            probability_yes=data.get("probability_yes", 0.5),
            probability_no=data.get("probability_no"),
            outcome_name=data.get("outcome_name", ""),
            consequence_type=ConsequenceType(data.get("consequence_type", "dispersion")),
            consequence_params=data.get("consequence_params", {}),
            is_terminal=data.get("is_terminal", False),
        )
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            node.add_child(child)
        return node


# ──────────────────────────────────────────────────────────────────────
# Event Tree Class
# ──────────────────────────────────────────────────────────────────────

class EventTree:
    """Event tree for QRA scenario analysis.

    Builds a tree from root initiating event through branching safety
    functions to terminal outcomes, calculating path probabilities
    and generating consequence scenarios.

    Examples
    --------
    >>> tree = EventTree("Vessel Leak", 5e-6)
    >>> tree.add_node("root", "Immediate Ignition?", prob_yes=0.1)
    >>> tree.add_node("ign_yes", "Explosion?", prob_yes=0.3)
    >>> # Mark terminal nodes with outcome types
    >>> tree.node("ign_yes_explosion_yes").set_terminal(
    ...     "VCE", ConsequenceType.EXPLOSION)
    >>> scenarios = tree.get_scenarios()
    >>> len(scenarios) > 0
    True
    """

    # Separator used for path building
    PATH_SEP: ClassVar[str] = "→"

    def __init__(
        self,
        name: str,
        initiating_frequency: float = 1.0,
        description: str = "",
    ) -> None:
        """Create an event tree.

        Parameters
        ----------
        name : str
            Name of the initiating event (e.g., "Pressurised Vessel Leak").
        initiating_frequency : float
            Initiating event frequency (events per year).
        description : str
            Description of the initiating event.
        """
        self.name = name
        self.initiating_frequency = initiating_frequency
        self.description = description

        # Root node represents the initiating event
        self._root = EventTreeNode(
            name=name,
            description=description or f"Initiating event: {name}",
            probability_yes=1.0,  # Initiating event has occurred
        )
        self._node_index: dict[str, EventTreeNode] = {"root": self._root}

    # ── Properties ───────────────────────────────────────────────────

    @property
    def root(self) -> EventTreeNode:
        """Get the root node."""
        return self._root

    @property
    def node_count(self) -> int:
        """Total number of nodes in the tree."""
        return len(self._node_index)

    # ── Node Management ──────────────────────────────────────────────

    def add_node(
        self,
        parent_name: str,
        name: str,
        description: str = "",
        prob_yes: float = 0.5,
        prob_no: Optional[float] = None,
        branch_type: Union[str, BranchType] = BranchType.YES_NO,
    ) -> tuple[EventTreeNode, EventTreeNode]:
        """Add a branching node with yes/no children.

        Each branching point creates TWO child nodes under the parent:
        - {name}_yes : success/yes branch
        - {name}_no  : failure/no branch

        Parameters
        ----------
        parent_name : str
            Name of parent node (or node path key) to attach children to.
        name : str
            Base name for the branching point.
        description : str
            Human-readable description.
        prob_yes : float
            Probability of the "yes" branch (success).
        prob_no : float, optional
            Probability of the "no" branch. Computed as 1-prob_yes if None.
        branch_type : str or BranchType
            Type of branch.

        Returns
        -------
        tuple of (EventTreeNode, EventTreeNode)
            (yes_child, no_child) nodes.
        """
        parent = self._find_node(parent_name)
        if parent is None:
            raise ValueError(f"Parent node '{parent_name}' not found")

        bt = BranchType(branch_type) if isinstance(branch_type, str) else branch_type

        yes_key = f"{name}_yes"
        no_key = f"{name}_no"

        yes_node = EventTreeNode(
            name=f"{name}: YES",
            description=f"{description} — Success (YES)",
            branch_type=bt,
            probability_yes=prob_yes,
        )
        no_node = EventTreeNode(
            name=f"{name}: NO",
            description=f"{description} — Failure (NO)",
            branch_type=bt,
            probability_yes=prob_no if prob_no is not None else (1.0 - prob_yes),
        )

        parent.add_child(yes_node)
        parent.add_child(no_node)

        self._node_index[yes_key] = yes_node
        self._node_index[no_key] = no_node

        return yes_node, no_node

    def add_terminal_node(
        self,
        parent_name: str,
        outcome_name: str,
        consequence_type: Union[str, ConsequenceType] = ConsequenceType.DISPERSION,
        description: str = "",
        consequence_params: Optional[dict[str, Any]] = None,
    ) -> EventTreeNode:
        """Add a terminal (leaf) outcome node.

        Parameters
        ----------
        parent_name : str
            Name of parent node.
        outcome_name : str
            Scenario name for this terminal outcome.
        consequence_type : str or ConsequenceType
            Type of consequence for this scenario.
        description : str
            Outcome description.
        consequence_params : dict, optional
            Additional parameters for consequence modelling.

        Returns
        -------
        EventTreeNode
            The new terminal node.
        """
        parent = self._find_node(parent_name)
        if parent is None:
            raise ValueError(f"Parent node '{parent_name}' not found")

        ct = ConsequenceType(consequence_type) if isinstance(consequence_type, str) else consequence_type

        node = EventTreeNode(
            name=f"Outcome: {outcome_name}",
            description=description or outcome_name,
            outcome_name=outcome_name,
            consequence_type=ct,
            consequence_params=consequence_params or {},
            is_terminal=True,
        )
        parent.add_child(node)
        self._node_index[outcome_name] = node
        return node

    def node(self, key: str) -> Optional[EventTreeNode]:
        """Get a node by its registered key name."""
        return self._node_index.get(key)

    def _find_node(self, name_or_key: str) -> Optional[EventTreeNode]:
        """Find a node by name, key, or partial match."""
        # Direct key lookup
        if name_or_key in self._node_index:
            return self._node_index[name_or_key]
        # Search by node.name
        for node in self._node_index.values():
            if node.name == name_or_key:
                return node
        # Try as a path like "ign_yes_explosion_yes"
        parts = name_or_key.split("_")
        for key, node in self._node_index.items():
            if key.endswith(name_or_key) or name_or_key.endswith(key):
                return node
        return None

    # ── Path Probability Calculation ─────────────────────────────────

    def _collect_paths(
        self,
        node: EventTreeNode,
        current_prob: float = 1.0,
        path_steps: Optional[list[str]] = None,
    ) -> list[tuple[EventTreeNode, float, list[str]]]:
        """Recursively collect all root-to-leaf paths with probabilities.

        Returns
        -------
        list of (terminal_node, cumulative_probability, path_steps)
        """
        if path_steps is None:
            path_steps = []

        results: list[tuple[EventTreeNode, float, list[str]]] = []

        if node.is_leaf:
            results.append((node, current_prob, list(path_steps)))
        else:
            for child in node.children:
                # The child's probability_yes carries the branch
                # probability for this specific path
                branch_prob = child.probability_yes
                new_prob = current_prob * branch_prob
                new_steps = path_steps + [child.name]
                results.extend(self._collect_paths(child, new_prob, new_steps))

        return results

    def calculate_path_probabilities(self) -> dict[str, float]:
        """Calculate probabilities for all terminal outcome paths.

        Returns
        -------
        dict
            Mapping of outcome name or path label to probability.
            Probabilities are already multiplied by the initiating
            event frequency.

        Notes
        -----
        The returned probabilities sum to (approximately) the
        initiating frequency × 1.0 (path probabilities sum to 1).

        Examples
        --------
        >>> tree = create_generic_vessel_tree()
        >>> probs = tree.calculate_path_probabilities()
        >>> abs(sum(probs.values()) - tree.initiating_frequency) < 1e-10
        True
        """
        paths = self._collect_paths(self._root)
        result: dict[str, float] = {}

        for node, path_prob, steps in paths:
            scenario_prob = path_prob * self.initiating_frequency
            label = node.outcome_name if node.outcome_name else self.PATH_SEP.join(steps)
            result[label] = scenario_prob

        return result

    def get_scenarios(self) -> list[Scenario]:
        """Extract all terminal outcome scenarios.

        Each scenario includes its probability (path probability ×
        initiating frequency), consequence type, and parameters.

        Returns
        -------
        list of Scenario
            All terminal outcome scenarios from the event tree.

        Examples
        --------
        >>> tree = create_generic_vessel_tree()
        >>> scenarios = tree.get_scenarios()
        >>> len(scenarios) > 0
        True
        >>> all(isinstance(s, Scenario) for s in scenarios)
        True
        """
        paths = self._collect_paths(self._root)
        scenarios: list[Scenario] = []

        for node, path_prob, steps in paths:
            scenario_prob = path_prob * self.initiating_frequency

            scenario = Scenario(
                name=node.outcome_name or self.PATH_SEP.join(steps),
                description=node.description,
                probability=scenario_prob,
                consequence_type=node.consequence_type,
                consequence_params=dict(node.consequence_params),
                path=steps,
            )
            scenarios.append(scenario)

        return scenarios

    def get_scenarios_by_type(
        self,
        consequence_type: Union[str, ConsequenceType],
    ) -> list[Scenario]:
        """Filter scenarios by consequence type."""
        ct = ConsequenceType(consequence_type) if isinstance(consequence_type, str) else consequence_type
        return [s for s in self.get_scenarios() if s.consequence_type == ct]

    # ── Tree Traversal ───────────────────────────────────────────────

    def traverse(self, callback, node: Optional[EventTreeNode] = None, depth: int = 0) -> None:
        """Depth-first traversal calling callback(node, depth) at each node."""
        if node is None:
            node = self._root
        callback(node, depth)
        for child in node.children:
            self.traverse(callback, child, depth + 1)

    def print_tree(self) -> str:
        """Generate an ASCII representation of the tree."""
        lines: list[str] = []

        def _print_node(node: EventTreeNode, depth: int) -> None:
            indent = "  " * depth
            prefix = "├─ " if depth > 0 else "■ "
            term = " [TERMINAL]" if node.is_terminal else ""
            prob_info = f" (P={node.probability_yes:.4f})" if depth > 0 else ""
            lines.append(f"{indent}{prefix}{node.name}{prob_info}{term}")
            for child in node.children:
                _print_node(child, depth + 1)

        _print_node(self._root, 0)
        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Validate tree structure and return list of issues.

        Checks:
        - No duplicate terminal names
        - Path probabilities sum to 1.0 (within tolerance)
        - All terminal nodes have assigned consequence types
        - Yes/no probabilities are valid (0 to 1)

        Returns
        -------
        list of str
            Validation warning/error messages. Empty = valid.
        """
        issues: list[str] = []

        # Check root has children
        if len(self._root.children) == 0:
            issues.append("Tree has no branches; add at least one branching node.")

        # Collect terminal nodes
        paths = self._collect_paths(self._root)
        if not paths:
            issues.append("No terminal outcomes found.")

        # Check path probability sum
        path_sum = sum(p for _, p, _ in paths)
        if abs(path_sum - 1.0) > 0.02:
            issues.append(
                f"Path probabilities sum to {path_sum:.6f}, "
                f"expected 1.0 (±2% tolerance)"
            )

        # Check duplicate terminal outcome names
        names: list[str] = []
        for node, _, _ in paths:
            if node.outcome_name:
                names.append(node.outcome_name)
        if len(names) != len(set(names)):
            issues.append("Duplicate terminal outcome names detected.")

        # Check probability validity
        def _check_prob(node: EventTreeNode) -> None:
            if node != self._root:
                if not (0.0 <= node.probability_yes <= 1.0):
                    issues.append(
                        f"Invalid probability {node.probability_yes} at node '{node.name}'"
                    )
            for child in node.children:
                _check_prob(child)

        _check_prob(self._root)

        return issues

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire event tree to a dictionary."""
        return {
            "name": self.name,
            "initiating_frequency": self.initiating_frequency,
            "description": self.description,
            "root": self._root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventTree":
        """Deserialize an event tree from a dictionary."""
        tree = cls(
            name=data.get("name", "Unnamed"),
            initiating_frequency=data.get("initiating_frequency", 1.0),
            description=data.get("description", ""),
        )
        if "root" in data:
            new_root = EventTreeNode.from_dict(data["root"])
            # Rebuild node index from the deserialized tree
            tree._root = new_root
            tree._rebuild_index()
        return tree

    def _rebuild_index(self) -> None:
        """Rebuild the node index after deserialization."""
        self._node_index = {}

        def _index(node: EventTreeNode) -> None:
            self._node_index[node.name] = node
            if node.outcome_name:
                self._node_index[node.outcome_name] = node
            for child in node.children:
                _index(child)

        _index(self._root)
        self._node_index["root"] = self._root

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "EventTree":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> "EventTree":
        """Deep copy of the event tree."""
        return EventTree.from_dict(self.to_dict())


# ──────────────────────────────────────────────────────────────────────
# Factory functions for common event trees
# ──────────────────────────────────────────────────────────────────────

def create_generic_vessel_tree(
    name: str = "Pressurised Vessel Leak",
    freq: float = 5e-6,
) -> EventTree:
    """Create a generic vessel release event tree.

    Branches:
      Initiating: Vessel leak
        → Immediate Ignition? (YES → jet fire / NO → continue)
          → YES: Terminal (jet fire)
          → NO: Delayed Ignition? (YES → continue / NO → safe)
            → YES: Explosion? (YES → explosion / NO → flash fire)
            → NO: Terminal (safe dispersal)

    Parameters
    ----------
    name : str
        Initiating event name.
    freq : float
        Initiating frequency (per year). Default 5e-6 (vessel full bore).

    Returns
    -------
    EventTree
        Pre-built generic vessel release event tree.
    """
    tree = EventTree(name, freq, f"Generic event tree for {name}")

    # Branch 1: Immediate ignition
    imm_yes, imm_no = tree.add_node(
        "root", "imm_ign",
        description="Immediate ignition of release",
        prob_yes=0.1,
        prob_no=0.9,
    )

    # Terminal: immediate ignition → jet fire
    imm_yes.is_terminal = True
    imm_yes.outcome_name = "Jet Fire"
    imm_yes.consequence_type = ConsequenceType.JET_FIRE
    imm_yes.consequence_params = {"fire_type": "jet_fire", "description": "Immediate ignition jet fire"}

    # Branch 2: Delayed ignition
    del_yes, del_no = tree.add_node(
        "imm_ign_no", "del_ign",
        description="Delayed ignition of vapour cloud",
        prob_yes=0.3,
        prob_no=0.7,
    )

    # Terminal: no delayed ignition → safe dispersal
    del_no.is_terminal = True
    del_no.outcome_name = "Safe Dispersal"
    del_no.consequence_type = ConsequenceType.SAFE_DISPERSAL
    del_no.consequence_params = {"description": "Vapour disperses safely, no ignition"}

    # Branch 3: Explosion vs flash fire
    exp_yes, exp_no = tree.add_node(
        "del_ign_yes", "explosion",
        description="Vapour cloud explosion (VCE)",
        prob_yes=0.4,
        prob_no=0.6,
    )

    # Terminal: explosion
    exp_yes.is_terminal = True
    exp_yes.outcome_name = "VCE Explosion"
    exp_yes.consequence_type = ConsequenceType.EXPLOSION
    exp_yes.consequence_params = {
        "explosion_type": "vce",
        "description": "Vapour cloud explosion",
        "yield_factor": 0.05,
    }

    # Terminal: flash fire
    exp_no.is_terminal = True
    exp_no.outcome_name = "Flash Fire"
    exp_no.consequence_type = ConsequenceType.FLASH_FIRE
    exp_no.consequence_params = {
        "fire_type": "flash_fire",
        "description": "Delayed ignition resulting in flash fire",
    }

    return tree


def create_generic_pipeline_tree(
    name: str = "Pipeline Rupture",
    freq: float = 1e-5,
    length_km: float = 1.0,
) -> EventTree:
    """Create a generic pipeline release event tree.

    Similar to vessel tree but with pipeline-specific initiating
    frequency scaling by length.

    Parameters
    ----------
    name : str
        Initiating event name.
    freq : float
        Base frequency per km-year. Default 1e-5 (full bore per km).
    length_km : float
        Pipeline segment length in km.

    Returns
    -------
    EventTree
        Pipeline release event tree.
    """
    total_freq = freq * length_km
    return create_generic_vessel_tree(
        name=f"{name} ({length_km} km)",
        freq=total_freq,
    )
