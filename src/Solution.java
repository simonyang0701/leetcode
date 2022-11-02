import java.util.*;
import lib.ListNode;
import lib.Printer;
import lib.TreeNode;

public class Solution {
    // No. 1
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[] { map.get(complement), i };
            }
            map.put(nums[i], i);
        }
        return null;
    }

    // No. 2
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode curr = dummyHead;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int x = (l1 != null) ? l1.val : 0;
            int y = (l2 != null) ? l2.val : 0;
            int sum = carry + x + y;
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
            if (l1 != null)
                l1 = l1.next;
            if (l2 != null)
                l2 = l2.next;
        }
        return dummyHead.next;
    }

    // No. 3
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> chars = new HashMap<>();

        int left = 0, right = 0;
        int res = 0;
        while(right < s.length()){
            char r = s.charAt(right);
            chars.put(r, chars.getOrDefault(r, 0) + 1);
            while(chars.get(r) > 1){
                char l = s.charAt(left);
                chars.put(l, chars.get(l) - 1);
                left ++;
            }
            res = Math.max(res, right - left + 1);

            right ++;
        }
        return res;
    }

    // No. 5
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++){
            int len1 = expandAroundCenter(s, i , i);
            int len2 = expandAroundCenter(s, i, i+1);
            int len = Math.max(len1, len2);
            if (len > end - start){
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }
    private int expandAroundCenter(String s, int left, int right){
        int L = left, R = right;
        while(L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)){
            L--;
            R++;
        }
        return R-L-1;
    }

    // No. 6
    public String convert(String s, int numRows) {
        if (numRows == 1) return s;
        List<StringBuilder> rows = new ArrayList<>();
        for(int i = 0; i < Math.min(numRows, s.length()); i++){
            rows.add(new StringBuilder());
        }
        int curRow = 0;
        boolean goingDown = false;
        for(char c: s.toCharArray()){
            rows.get(curRow).append(c);
            if (curRow == 0 || curRow == numRows - 1)
                goingDown = !goingDown;
            curRow += goingDown ? 1 : -1;
        }
        StringBuilder ret = new StringBuilder();
        for(StringBuilder row: rows) ret.append(row);
        return ret.toString();
    }

    // No. 7
    public int reverse(int x) {
        if (x == 0) return x;
        int num = 0;
        while(x != 0){
            int digit = x % 10;
            x /= 10;
            if (num > 0 && (Integer.MAX_VALUE - digit) / 10 < num) return 0;
            if (num < 0 && (Integer.MAX_VALUE - digit) / 10 > num) return 0;

            num = num * 10 + digit;
        }
        return num;

    }

    // No. 34
    public int[] searchRange(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi){
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target && nums[lo] == nums[hi] ){
                return new int[] {lo, hi};
            } else if(nums[mid] == target && nums[lo] < target){
                lo ++;
            } else if(nums[mid] == target && target < nums[hi]){
                hi --;
            } else if(nums[mid] < target){
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return new int[] {-1, -1};
    }

    // No. 78
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums.length == 0) return result;
        Arrays.sort(nums);
        dfsSubsets(nums, 0, new ArrayList<>(), result);
        return result;
    }
    private void dfsSubsets(int[] nums, int index, List<Integer> path, List<List<Integer>> result){
        result.add(new ArrayList<>(path));
        for (int i = index; i < nums.length; i++){
            path.add(nums[i]);
            dfsSubsets(nums, i+1, path, result);
            path.remove(path.size() - 1);
        }
    }

    // No. 104
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // No. 139
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    // No. 146
    static class LRUCache {
        private Map<Integer, Integer> map = new LinkedHashMap<>();
        private int capacity;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            map = new LinkedHashMap<>();
        }

        public int get(int key) {
            if (!map.containsKey(key)) return -1;
            int val = map.get(key);
            put(key, val);
            return val;
        }

        public void put(int key, int value) {
            if(!map.containsKey(key) && map.size() == capacity){
                map.remove(map.keySet().iterator().next());
            }
            map.remove(key);
            map.put(key, value);
        }
    }

    // No. 199
    public List<Integer> rightSideView(TreeNode root) {
        if(root == null) return new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        List<Integer> rightside = new ArrayList<>();
        while(!queue.isEmpty()){
            int level = queue.size();
            for(int i = 0; i < level; i++){
                TreeNode node = queue.poll();
                if (i == level - 1){
                    rightside.add((Integer) node.val);
                }
                if(node.left != null) {
                    queue.offer(node.left);
                }
                if(node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return rightside;
    }

    // No. 261
    public boolean validTree(int n, int[][] edges) {
        if (n < 1) return false;
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; i++) map.put(i, new HashSet<>());
        for (int[] edge: edges){
            map.get(edge[0]).add(edge[1]);
            map.get(edge[1]).add(edge[0]);
        }

        Set<Integer> set = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.add(0);
        while(!queue.isEmpty()){
            int top = queue.remove();
            if (set.contains(top)) return false;

            for(int node: map.get(top)){
                queue.add(node);
                map.get(node).remove(top);
            }
            set.add(top);
        }
        return set.size() == n;
    }

    // No. 209
    public int minSubArrayLen(int target, int[] nums) {
        /*
        Two points
        Time complexity: O(n)
        Space complexity: O(1)
         */
        int n = nums.length, ans = Integer.MAX_VALUE, left = 0, right = 0;
        int sum = 0;
        for (int i = 0; i < n; i++){
            sum += nums[i];
            while(sum >= target){
                ans = Math.min(ans, i + 1 - left);
                sum -= nums[left++];
            }
        }
        return (ans != Integer.MAX_VALUE) ? ans: 0;
    }

    // No. 311
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        int m = mat1.length, n = mat1[0].length, n2 = mat2[0].length;
        int[][] ret = new int[m][n2];
        for(int i = 0; i < m; i++){
            for (int k = 0;k<n;k++){
                if(mat1[i][k] != 0){
                    for (int j = 0; j<n2; j++){
                        if(mat2[k][j] != 0){
                            ret[i][j] += mat1[i][k] * mat2[k][j];
                        }
                    }
                }
            }
        }
        return ret;
    }

    // No. 981
    static class TimeMap {
        /*
        Time complexity: set() O(L*M*logM)
                         get() O(N*(L*logM + logM))
        Space complexity: set() O(M*L)
         */
        HashMap<String, TreeMap<Integer, String>> keyTimeMap;

        public TimeMap() {
            keyTimeMap = new HashMap<String, TreeMap<Integer, String>>();
        }

        public void set(String key, String value, int timestamp) {
            if (!keyTimeMap.containsKey(key)) {
                keyTimeMap.put(key, new TreeMap<Integer, String>());
            }

            keyTimeMap.get(key).put(timestamp, value);
        }

        public String get(String key, int timestamp) {
            if (!keyTimeMap.containsKey(key)) {
                return "";
            }

            Integer floorKey = keyTimeMap.get(key).floorKey(timestamp);
            if (floorKey != null) {
                return keyTimeMap.get(key).get(floorKey);
            }

            return "";
        }
    }

    // No. 2290
    public int minMutation(String start, String end, String[] bank) {
        /*
        Time complexity: O(B)
        Space complexity: O(1)
         */
        Queue<String> queue = new LinkedList<>();
        Set<String> seen = new HashSet<>();
        queue.add(start);
        seen.add(start);

        int steps = 0;
        while(!queue.isEmpty()){
            int nodeInQueue = queue.size();
            for (int j = 0; j < nodeInQueue; j++){
                String node = queue.remove();
                if(node.equals(end)) return steps;
                for (char c: new char[]{'A', 'C', 'G', 'T'}){
                    for (int i = 0; i < node.length(); i++){
                        String neighbor = node.substring(0, i) + c + node.substring(i+1);
                        if(!seen.contains(neighbor) && Arrays.asList(bank).contains(neighbor)){
                            queue.add(neighbor);
                            seen.add(neighbor);
                        }
                    }
                }
            }
            steps ++;
        }
        return -1;
    }
}
