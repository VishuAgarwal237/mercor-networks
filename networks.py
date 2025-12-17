"""
Referral Network Challenge - Complete Implementation
"""

from typing import Iterable, Optional, Callable  # type hints for function signatures
from collections import defaultdict, deque  # defaultdict for graph, deque for BFS


# =============================================================================
# Part 1 — Graph
# =============================================================================

class ReferralError(ValueError):  # custom exception inheriting from ValueError
    """Raised when a referral operation would violate graph constraints."""
    pass  # no additional functionality needed


class ReferralNetwork:
    """
    A directed graph where edges represent referrer → candidate relationships.
    Invariants: no self-referrals, each candidate has at most one referrer, acyclic.
    """
    
    def __init__(self):  # initialize empty graph
        self._children: dict[str, set[str]] = defaultdict(set)  # referrer -> set of candidates they referred
        self._parent: dict[str, str] = {}  # candidate -> their referrer (at most one)
    
    def add_referral(self, referrer: str, candidate: str) -> None:  # add edge referrer -> candidate
        """Add a referral edge. Raises ReferralError if it would violate constraints."""
        
        if referrer == candidate:  # check for self-referral
            raise ReferralError("Self-referral not allowed")  # rule 1 violated
        
        if candidate in self._parent:  # check if candidate already has a referrer
            raise ReferralError("Candidate already has a referrer")  # rule 2 violated
        
        if self._would_create_cycle(referrer, candidate):  # check for cycle
            raise ReferralError("Would create a cycle")  # rule 3 violated
        
        # All checks passed, add the edge atomically
        self._children[referrer].add(candidate)  # add candidate to referrer's children
        self._parent[candidate] = referrer  # record referrer as candidate's parent
    
    def _would_create_cycle(self, referrer: str, candidate: str) -> bool:  # cycle detection
        """Check if adding edge referrer->candidate would create a cycle."""
        # A cycle occurs if candidate is an ancestor of referrer
        # i.e., if we can reach candidate by following parent pointers from referrer
        current = referrer  # start from the referrer
        while current in self._parent:  # walk up the ancestor chain
            current = self._parent[current]  # move to parent
            if current == candidate:  # found candidate in ancestor chain
                return True  # cycle would be created
        return False  # no cycle detected
    
    def direct_referrals(self, user: str) -> Iterable[str]:  # immediate children
        """Return the immediate candidates referred by user."""
        return iter(self._children.get(user, set()))  # return iterator over children set
    
    def all_referrals(self, user: str) -> Iterable[str]:  # all descendants
        """Return all direct and indirect referrals (descendants) of user."""
        result = []  # collect all descendants
        queue = deque(self._children.get(user, set()))  # BFS queue starting with direct children
        visited = set()  # track visited to avoid duplicates (shouldn't happen in DAG but safe)
        
        while queue:  # BFS traversal
            current = queue.popleft()  # get next node
            if current not in visited:  # skip if already visited
                visited.add(current)  # mark as visited
                result.append(current)  # add to result
                queue.extend(self._children.get(current, set()))  # add current's children to queue
        
        return result  # return all descendants
    
    def get_all_users(self) -> set[str]:  # helper to get all nodes in graph
        """Return all users in the network."""
        users = set(self._children.keys())  # all referrers
        users.update(self._parent.keys())  # all candidates (might not be referrers themselves)
        return users  # combined set of all users


# =============================================================================
# Part 2 — Influence
# =============================================================================

def top_k_by_reach(network: ReferralNetwork, k: int) -> list[str]:  # rank by descendant count
    """Return top k users ranked by number of descendants (reach)."""
    users = network.get_all_users()  # get all users in network
    
    reach_scores = []  # list of (user, reach_count) tuples
    for user in users:  # compute reach for each user
        descendants = list(network.all_referrals(user))  # get all descendants
        reach_scores.append((user, len(descendants)))  # store user and count
    
    reach_scores.sort(key=lambda x: (-x[1], x[0]))  # sort by reach desc, then name asc for ties
    
    return [user for user, _ in reach_scores[:k]]  # return top k users


def top_k_by_flow_centrality(network: ReferralNetwork, k: int) -> list[str]:  # betweenness centrality variant
    """
    Return top k users ranked by flow centrality.
    Flow centrality(u): count of ordered pairs (s,t) where s≠u≠t and u lies on
    a shortest path from s to t. Endpoints don't count as on the path.
    """
    users = list(network.get_all_users())  # get all users as list
    n = len(users)  # number of users
    
    if n == 0:  # empty network
        return []  # no users to rank
    
    # Build adjacency list for BFS
    adj: dict[str, list[str]] = defaultdict(list)  # adjacency list representation
    for user in users:  # for each user
        for child in network.direct_referrals(user):  # for each direct referral
            adj[user].append(child)  # add directed edge
    
    centrality: dict[str, int] = defaultdict(int)  # flow centrality scores
    
    # For each source, find shortest paths to all reachable nodes
    for s in users:  # iterate over all sources
        # BFS to find shortest paths from s
        dist: dict[str, int] = {s: 0}  # distance from s
        parent_lists: dict[str, list[str]] = defaultdict(list)  # predecessors on shortest paths
        queue = deque([s])  # BFS queue
        
        while queue:  # BFS traversal
            u = queue.popleft()  # current node
            for v in adj[u]:  # for each neighbor
                if v not in dist:  # first time reaching v
                    dist[v] = dist[u] + 1  # record distance
                    parent_lists[v].append(u)  # u is a predecessor of v
                    queue.append(v)  # add v to queue
                elif dist[v] == dist[u] + 1:  # another shortest path to v
                    parent_lists[v].append(u)  # add u as additional predecessor
        
        # For each target t reachable from s, count intermediates on shortest paths
        for t in dist:  # for each reachable target
            if t == s:  # skip source itself
                continue  # no path from s to s counts
            
            # Backtrack from t to s, collecting all nodes on shortest paths
            intermediates = set()  # nodes between s and t on shortest paths
            visited_backtrack = set()  # avoid revisiting during backtrack
            backtrack_queue = deque([t])  # start from target
            
            while backtrack_queue:  # backtrack BFS
                node = backtrack_queue.popleft()  # current node
                if node in visited_backtrack:  # already processed
                    continue  # skip
                visited_backtrack.add(node)  # mark as processed
                
                for pred in parent_lists[node]:  # for each predecessor
                    if pred != s:  # don't count source as intermediate
                        intermediates.add(pred)  # pred is on a shortest path
                        backtrack_queue.append(pred)  # continue backtracking
            
            # Each intermediate on shortest path s->t gets +1
            for u in intermediates:  # for each intermediate node
                centrality[u] += 1  # increment flow centrality
    
    # Sort by centrality descending, then by name ascending for ties
    sorted_users = sorted(users, key=lambda u: (-centrality[u], u))  # stable sort
    
    return sorted_users[:k]  # return top k


# =============================================================================
# Part 3 — Growth
# =============================================================================

def expected_network_size(p: float, days: int) -> float:  # expected growth model
    """
    Model expected network growth over discrete days.
    - Each referrer starts with capacity 10 successful referrals
    - On any day, an active referrer makes at most 1 referral with probability p
    - Success consumes 1 capacity; at 0 capacity, referrer becomes inactive
    - New referrals join next day with full capacity
    - Day 0 starts with 100 active referrers
    Returns expected network size at end of given days.
    """
    # Track expected count of referrers at each capacity level (1 to 10)
    # capacity[c] = expected number of referrers with c remaining capacity
    capacity = [0.0] * 11  # index 0 unused, indices 1-10 for capacity levels
    capacity[10] = 100.0  # start with 100 referrers at capacity 10
    
    total_referrals = 0.0  # cumulative expected successful referrals
    
    for day in range(days + 1):  # simulate days 0 through 'days' inclusive
        # Each referrer with capacity c has probability p of success
        # Success: c decreases by 1, one new referral joins next day
        # Failure: c stays the same
        
        new_capacity = [0.0] * 11  # capacity distribution for next day
        day_referrals = 0.0  # expected referrals made this day
        
        for c in range(1, 11):  # for each capacity level
            count = capacity[c]  # expected referrers at this capacity
            if count == 0:  # no referrers at this level
                continue  # skip
            
            # Expected referrers who succeed (make a referral)
            succeed = count * p  # expected successes
            fail = count * (1 - p)  # expected failures
            
            day_referrals += succeed  # add to day's referral count
            
            if c > 1:  # if capacity > 1, successful referrers move to c-1
                new_capacity[c - 1] += succeed  # capacity decreases
            # if c == 1, successful referrers drop to 0 capacity (inactive)
            
            new_capacity[c] += fail  # failed referrers stay at same capacity
        
        total_referrals += day_referrals  # accumulate total referrals
        
        # New referrals from this day join tomorrow with capacity 10
        new_capacity[10] += day_referrals  # they join next day
        
        capacity = new_capacity  # update for next day
    
    return 100.0 + total_referrals  # initial 100 plus all successful referrals


# =============================================================================
# Part 4 — Incentive
# =============================================================================

def min_bonus_for_target(
    days: int,  # number of days to run
    target_network_size: int,  # target size to reach
    adoption_prob: Callable[[int], float]  # black-box function: bonus -> probability
) -> Optional[int]:  # return smallest valid bonus or None
    """
    Find minimum bonus (in $10 increments) to reach target network size.
    adoption_prob(bonus) returns participation probability for given bonus.
    Returns None if no bonus can reach the target.
    """
    # Binary search for minimum bonus
    # Since adoption_prob is monotonically non-decreasing,
    # expected_network_size is also monotonically non-decreasing in bonus
    
    # First, find an upper bound where target is achievable
    # Also check if target is achievable at all (even with p=1)
    max_p = adoption_prob(10000)  # check at high bonus
    max_size = expected_network_size(max_p, days)  # max achievable size
    
    if max_size < target_network_size:  # even max probability can't reach target
        # Try even higher to confirm
        test_bonus = 10000  # start high
        while test_bonus <= 1000000:  # reasonable upper limit
            p = adoption_prob(test_bonus)  # get probability
            size = expected_network_size(p, days)  # compute expected size
            if size >= target_network_size:  # found achievable bonus
                break  # exit search for upper bound
            test_bonus += 10000  # increase bonus
        else:  # loop completed without finding
            return None  # target not achievable
    
    # Binary search between 0 and upper bound
    lo = 0  # minimum bonus
    hi = 1000000  # generous upper bound
    
    # First check if bonus 0 already works
    p0 = adoption_prob(0)  # probability at zero bonus
    if expected_network_size(p0, days) >= target_network_size:  # target met at 0
        return 0  # no bonus needed
    
    result = None  # track best valid bonus found
    
    # Binary search for minimum bonus
    while lo <= hi:  # standard binary search
        mid = ((lo + hi) // 2 // 10) * 10  # round to nearest $10
        if mid < lo:  # edge case at low values
            mid = lo  # use lo instead
        
        p = adoption_prob(mid)  # get probability at mid bonus
        size = expected_network_size(p, days)  # compute expected size
        
        if size >= target_network_size:  # target met
            result = mid  # record this bonus
            hi = mid - 10  # search for smaller bonus
        else:  # target not met
            lo = mid + 10  # need higher bonus
    
    return result  # return minimum bonus found, or None


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":  # only run when executed directly
    
    # Test Part 1 - Graph
    print("=== Part 1: Graph Tests ===")  # header
    net = ReferralNetwork()  # create network
    
    # Add some referrals
    net.add_referral("Alice", "Bob")  # Alice referred Bob
    net.add_referral("Alice", "Charlie")  # Alice referred Charlie
    net.add_referral("Bob", "David")  # Bob referred David
    net.add_referral("Charlie", "Eve")  # Charlie referred Eve
    net.add_referral("David", "Frank")  # David referred Frank
    
    print(f"Direct referrals of Alice: {list(net.direct_referrals('Alice'))}")  # should be Bob, Charlie
    print(f"All referrals of Alice: {list(net.all_referrals('Alice'))}")  # all descendants
    
    # Test constraints
    try:  # test self-referral
        net.add_referral("X", "X")  # should fail
    except ReferralError as e:  # expected
        print(f"Self-referral blocked: {e}")  # confirmed
    
    try:  # test duplicate referrer
        net.add_referral("Eve", "Bob")  # Bob already has referrer
    except ReferralError as e:  # expected
        print(f"Duplicate referrer blocked: {e}")  # confirmed
    
    try:  # test cycle
        net.add_referral("Frank", "Alice")  # would create cycle
    except ReferralError as e:  # expected
        print(f"Cycle blocked: {e}")  # confirmed
    
    # Test Part 2 - Influence
    print("\n=== Part 2: Influence Tests ===")  # header
    print(f"Top 3 by reach: {top_k_by_reach(net, 3)}")  # users with most descendants
    print(f"Top 3 by flow centrality: {top_k_by_flow_centrality(net, 3)}")  # users on most paths
    
    # Test Part 3 - Growth
    print("\n=== Part 3: Growth Tests ===")  # header
    size = expected_network_size(0.1, 30)  # 10% daily probability, 30 days
    print(f"Expected network size (p=0.1, days=30): {size:.2f}")  # print result
    
    size = expected_network_size(0.5, 30)  # 50% daily probability, 30 days
    print(f"Expected network size (p=0.5, days=30): {size:.2f}")  # print result
    
    # Test Part 4 - Incentive
    print("\n=== Part 4: Incentive Tests ===")  # header
    
    def sample_adoption(bonus: int) -> float:  # sample adoption probability function
        """Sample adoption probability: increases with bonus."""
        if bonus <= 0:  # no bonus
            return 0.05  # 5% base participation
        elif bonus <= 50:  # low bonus
            return 0.05 + bonus * 0.005  # linear increase
        elif bonus <= 200:  # medium bonus
            return 0.30 + (bonus - 50) * 0.003  # slower increase
        else:  # high bonus
            return min(0.75 + (bonus - 200) * 0.001, 1.0)  # cap at 1.0
    
    bonus = min_bonus_for_target(30, 200, sample_adoption)  # find min bonus for 200 users
    print(f"Min bonus for 200 users in 30 days: ${bonus}")  # print result
    
    bonus = min_bonus_for_target(30, 500, sample_adoption)  # find min bonus for 500 users
    print(f"Min bonus for 500 users in 30 days: ${bonus}")  # print result
    
    print("\n=== All tests completed ===")  # footer