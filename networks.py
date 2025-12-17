"""
Referral Network Challenge - Complete Implementation
"""

from typing import Iterable, Optional, Callable
from collections import defaultdict, deque


class ReferralError(ValueError):
    """Raised when a referral operation would violate graph constraints."""
    pass


class ReferralNetwork:
    """
    A directed graph where edges represent referrer → candidate relationships.
    Invariants: no self-referrals, each candidate has at most one referrer, acyclic.
    """
    
    def __init__(self):
        self._children: dict[str, set[str]] = defaultdict(set)
        self._parent: dict[str, str] = {}
    
    def add_referral(self, referrer: str, candidate: str) -> None:
        """Add a referral edge. Raises ReferralError if it would violate constraints."""
        if referrer == candidate:
            raise ReferralError("Self-referral not allowed")
        
        if candidate in self._parent:
            raise ReferralError("Candidate already has a referrer")
        
        if self._would_create_cycle(referrer, candidate):
            raise ReferralError("Would create a cycle")
        
        self._children[referrer].add(candidate)
        self._parent[candidate] = referrer
    
    def _would_create_cycle(self, referrer: str, candidate: str) -> bool:
        """Check if adding edge referrer->candidate would create a cycle."""
        current = referrer
        while current in self._parent:
            current = self._parent[current]
            if current == candidate:
                return True
        return False
    
    def direct_referrals(self, user: str) -> Iterable[str]:
        """Return the immediate candidates referred by user."""
        return self._children.get(user, set())
    
    def all_referrals(self, user: str) -> Iterable[str]:
        """Return all direct and indirect referrals (descendants) of user."""
        result = []
        queue = deque(self._children.get(user, set()))
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                result.append(current)
                queue.extend(self._children.get(current, set()))
        
        return result
    
    def get_all_users(self) -> set[str]:
        """Return all users in the network."""
        users = set(self._children.keys())
        users.update(self._parent.keys())
        return users


def top_k_by_reach(network: ReferralNetwork, k: int) -> list[str]:
    """Return top k users ranked by number of descendants (reach)."""
    users = network.get_all_users()
    
    reach_scores = []
    for user in users:
        descendants = list(network.all_referrals(user))
        reach_scores.append((user, len(descendants)))
    
    reach_scores.sort(key=lambda x: (-x[1], x[0]))
    
    return [user for user, _ in reach_scores[:k]]


def top_k_by_flow_centrality(network: ReferralNetwork, k: int) -> list[str]:
    """
    Return top k users ranked by flow centrality.
    Flow centrality(u): count of ordered pairs (s,t) where s≠u≠t and u lies on
    a shortest path from s to t. Endpoints don't count as on the path.
    """
    users = list(network.get_all_users())
    n = len(users)
    
    if n == 0:
        return []
    
    adj: dict[str, list[str]] = defaultdict(list)
    for user in users:
        for child in network.direct_referrals(user):
            adj[user].append(child)
    
    centrality: dict[str, int] = defaultdict(int)
    
    for s in users:
        dist: dict[str, int] = {s: 0}
        parent_lists: dict[str, list[str]] = defaultdict(list)
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent_lists[v].append(u)
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    parent_lists[v].append(u)
        
        for t in dist:
            if t == s:
                continue
            
            intermediates = set()
            visited_backtrack = set()
            backtrack_queue = deque([t])
            
            while backtrack_queue:
                node = backtrack_queue.popleft()
                if node in visited_backtrack:
                    continue
                visited_backtrack.add(node)
                
                for pred in parent_lists[node]:
                    if pred != s:
                        intermediates.add(pred)
                        backtrack_queue.append(pred)
            
            for u in intermediates:
                centrality[u] += 1
    
    sorted_users = sorted(users, key=lambda u: (-centrality[u], u))
    
    return sorted_users[:k]


def expected_network_size(p: float, days: int) -> float:
    """
    Model expected network growth over discrete days.
    - Each referrer starts with capacity 10 successful referrals
    - On any day, an active referrer makes at most 1 referral with probability p
    - Success consumes 1 capacity; at 0 capacity, referrer becomes inactive
    - New referrals join next day with full capacity
    - Day 0 starts with 100 active referrers
    Returns expected network size at end of given days.
    """
    capacity = [0.0] * 11
    capacity[10] = 100.0
    
    total_referrals = 0.0
    
    for day in range(days + 1):
        new_capacity = [0.0] * 11
        day_referrals = 0.0
        
        for c in range(1, 11):
            count = capacity[c]
            if count == 0:
                continue
            
            succeed = count * p
            fail = count * (1 - p)
            
            day_referrals += succeed
            
            if c > 1:
                new_capacity[c - 1] += succeed
            
            new_capacity[c] += fail
        
        total_referrals += day_referrals
        new_capacity[10] += day_referrals
        capacity = new_capacity
    
    return 100.0 + total_referrals


def min_bonus_for_target(
    days: int,
    target_network_size: int,
    adoption_prob: Callable[[int], float]
) -> Optional[int]:
    """
    Find minimum bonus (in $10 increments) to reach target network size.
    adoption_prob(bonus) returns participation probability for given bonus.
    Returns None if no bonus can reach the target.
    """
    max_p = adoption_prob(10000)
    max_size = expected_network_size(max_p, days)
    
    if max_size < target_network_size:
        test_bonus = 10000
        while test_bonus <= 1000000:
            p = adoption_prob(test_bonus)
            size = expected_network_size(p, days)
            if size >= target_network_size:
                break
            test_bonus += 10000
        else:
            return None
    
    lo = 0
    hi = 1000000
    
    p0 = adoption_prob(0)
    if expected_network_size(p0, days) >= target_network_size:
        return 0
    
    result = None
    
    while lo <= hi:
        mid = ((lo + hi) // 2 // 10) * 10
        if mid < lo:
            mid = lo
        
        p = adoption_prob(mid)
        size = expected_network_size(p, days)
        
        if size >= target_network_size:
            result = mid
            hi = mid - 10
        else:
            lo = mid + 10
    
    return result


if __name__ == "__main__":
    
    print("=== Part 1: Graph Tests ===")
    net = ReferralNetwork()
    
    net.add_referral("Alice", "Bob")
    net.add_referral("Alice", "Charlie")
    net.add_referral("Bob", "David")
    net.add_referral("Charlie", "Eve")
    net.add_referral("David", "Frank")
    
    print(f"Direct referrals of Alice: {list(net.direct_referrals('Alice'))}")
    print(f"All referrals of Alice: {list(net.all_referrals('Alice'))}")
    
    try:
        net.add_referral("X", "X")
    except ReferralError as e:
        print(f"Self-referral blocked: {e}")
    
    try:
        net.add_referral("Eve", "Bob")
    except ReferralError as e:
        print(f"Duplicate referrer blocked: {e}")
    
    try:
        net.add_referral("Frank", "Alice")
    except ReferralError as e:
        print(f"Cycle blocked: {e}")
    
    print("\n=== Part 2: Influence Tests ===")
    print(f"Top 3 by reach: {top_k_by_reach(net, 3)}")
    print(f"Top 3 by flow centrality: {top_k_by_flow_centrality(net, 3)}")
    
    print("\n=== Part 3: Growth Tests ===")
    size = expected_network_size(0.1, 30)
    print(f"Expected network size (p=0.1, days=30): {size:.2f}")
    
    size = expected_network_size(0.5, 30)
    print(f"Expected network size (p=0.5, days=30): {size:.2f}")
    
    print("\n=== Part 4: Incentive Tests ===")
    
    def sample_adoption(bonus: int) -> float:
        """Sample adoption probability: increases with bonus."""
        if bonus <= 0:
            return 0.05
        elif bonus <= 50:
            return 0.05 + bonus * 0.005
        elif bonus <= 200:
            return 0.30 + (bonus - 50) * 0.003
        else:
            return min(0.75 + (bonus - 200) * 0.001, 1.0)
    
    bonus = min_bonus_for_target(30, 200, sample_adoption)
    print(f"Min bonus for 200 users in 30 days: ${bonus}")
    
    bonus = min_bonus_for_target(30, 500, sample_adoption)
    print(f"Min bonus for 500 users in 30 days: ${bonus}")
    
    print("\n=== All tests completed ===")