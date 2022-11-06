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

    // No. 8
    public int myAtoi(String s) {
        if (s == null || s.length() == 0) return 0;
        s = s.trim();
        char firstChar = s.charAt(0);
        int sign = 1, start = 0, len = s.length();
        long sum = 0;
        if(firstChar == '+'){
            sign = 1;
            start ++;
        } else if (firstChar == '-'){
            sign = -1;
            start ++;
        }
        for (int i = start; i < len; i++){
            if (!Character.isDigit(s.charAt(i)))
                return (int) sum * sign;
            sum = sum * 10 + s.charAt(i) - '0';
            if (sign == 1 && sum > Integer.MAX_VALUE)
                return Integer.MAX_VALUE;
            if (sign == -1 && sum < Integer.MIN_VALUE)
                return Integer.MIN_VALUE;
        }

        return (int) sum * sign;
    }

    // No. 11
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, maxarea = 0;
        while(left < right){
            int width = right - left;
            maxarea = Math.max(maxarea, Math.min(height[left], height[right]) * width);
            if (height[left] <= height[right]){
                left ++;
            }else{
                right --;
            }
        }
        return maxarea;
    }

    // No. 12
    private static final int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    private static final String[] symbols = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};

    public String intToRoman(int num) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length && num > 0; i++) {
            while (values[i] <= num) {
                num -= values[i];
                sb.append(symbols[i]);
            }
        }
        return sb.toString();
    }

    // No. 15
    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> res = new HashSet<>();
        if (nums.length == 0) return new ArrayList<>(res);
        Arrays.sort(nums);
        for (int i = 0; i<nums.length - 2; i++){
            int j = i + 1;
            int k = nums.length - 1;
            while(j < k){
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0){
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                    k--;
                }
                else if (sum > 0) k --;
                else if (sum < 0) j ++;
            }
        }

        return new ArrayList<>(res);
    }

    // No. 16
    public int threeSumClosest(int[] nums, int target) {
        int diff = Integer.MAX_VALUE;
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < n && diff != 0; ++i) {
            int lo = i + 1;
            int hi = n - 1;
            while (lo < hi) {
                int sum = nums[i] + nums[lo] + nums[hi];
                if (Math.abs(target - sum) < Math.abs(diff)) {
                    diff = target - sum;
                }
                if (sum < target) {
                    ++lo;
                } else {
                    --hi;
                }
            }
        }
        return target - diff;
    }

    // No. 17
    public List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<>();
        if(digits.isEmpty()) return ans;
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++){
            int x = Character.getNumericValue(digits.charAt(i));
            while(ans.peek().length() == i){
                String t = ans.remove();
                for (char s : mapping[x].toCharArray()){
                    ans.add(t+s);
                }
            }
        }
        return ans;
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

    // No. 45
    public int jump(int[] nums) {
        int farest = 0, currentJumEnd = 0, res = 0;
        for (int i = 0; i < nums.length - 1; i++){
            farest = Math.max(farest, i + nums[i]);
            if (i == currentJumEnd){
                res ++;
                currentJumEnd = farest;
            }
        }
        return res;
    }

    // No. 53
    public int maxSubArray(int[] nums) {
        int globalMax = Integer.MIN_VALUE;
        int localMax = 0;
        for (int i = 0; i<nums.length; i++){
            if (localMax < 0){
                localMax = nums[i];
            }else{
                localMax = localMax + nums[i];
            }
            globalMax = Math.max(globalMax, localMax);
        }
        return globalMax;
    }

    // No. 56
    public int[][] merge(int[][] intervals) {
        /*
        Time complexity: O(nlogN)
        Space complexity: O(logN)
         */
        if (intervals.length <= 1) return intervals;
        Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1[0], i2[0]));

        List<int[]> result = new ArrayList<>();
        int[] newInterval = intervals[0];
        result.add(newInterval);
        for (int[] interval: intervals){
            if (interval[0] < newInterval[1]){
                newInterval[1] = Math.max(newInterval[1], interval[1]);
            }else{
                newInterval = interval;
                result.add(newInterval);
            }
        }

        return result.toArray(new int[result.size()][]);
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

    // No. 98
    public boolean isValidBST(TreeNode root) {
        return dfsisValidBST(root, null, null);
    }
    private boolean dfsisValidBST(TreeNode root, Integer low, Integer high){
        if (root == null) return true;
        if ((low != null && root.val <= low) || (high != null && root.val >= high)) return false;
        boolean left = dfsisValidBST(root.left, low, root.val);
        boolean right = dfsisValidBST(root.right, root.val, high);
        return left & right;
    }

    // No. 102
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while(!q.isEmpty()){
            int size = q.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                level.add((Integer) cur.val);
                if(cur.left != null) q.offer(cur.left);
                if(cur.right != null) q.offer(cur.right);
            }
            res.add(level);
        }
        return res;
    }

    // No. 103
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        Boolean even = true;
        while(!queue.isEmpty()){
            int size = queue.size();
            LinkedList<Integer> path = new LinkedList<>();
            for (int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                if (cur.left != null) queue.add(cur.left);
                if (cur.right != null) queue.add(cur.right);
                if (even){
                    path.add((Integer) cur.val);
                }else{
                    path.addFirst((Integer) cur.val);
                }
            }
            res.add(path);
            even = !even;
        }
        return res;
    }

    // No. 104
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // No. 107
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        LinkedList<List<Integer>> res = new LinkedList<>();
        if(root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> path = new ArrayList<>();
            for (int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                path.add(cur.val);
                if (cur.left != null) queue.offer(cur.left);
                if (cur.right != null) queue.offer(cur.right);
            }
            res.addFirst(path);
        }
        List<List<Integer>> ret = new ArrayList<>();
        for(List<Integer> item: res){
            ret.add(item);
        }
        return ret;
    }

    // No. 100
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val == q.val){
            boolean left = isSameTree(p.left, q.left);
            boolean right = isSameTree(p.right, q.right);
            return (left & right);
        }
        return false;
    }

    // No. 113
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        dfspathSum(root, targetSum, res, new ArrayList<>(), 0);
        return res;
    }
    private void dfspathSum(TreeNode root, int targetSum, List<List<Integer>> res, List<Integer> path, int sum){
        if (root == null) return;
        path.add(root.val);
        sum += root.val;
        if (sum == targetSum && root.left == null && root.right == null){
            res.add(new ArrayList<>(path));
        }
        dfspathSum(root.left, targetSum, res, path, sum);
        dfspathSum(root.right, targetSum, res, path, sum);
        path.remove(path.size() - 1);
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

    // No. 148
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode mid = getMid(head);
        ListNode l1, l2;
        l1 = head;
        l2 = mid.next;
        mid.next = null;
        l1 = sortList(l1);
        l2 = sortList(l2);
        return merge(l1, l2);

    }
    ListNode merge(ListNode l1, ListNode l2){
        ListNode dummy = new ListNode();
        ListNode tail = dummy;
        while(l1 != null && l2 != null){
            if (l1.val <= l2.val){
                tail.next = l1;
                l1 = l1.next;
            }else{
                tail.next = l2;
                l2 = l2.next;
            }
            tail = tail.next;
        }
        tail.next = (l1 != null) ? l1: l2;
        return dummy.next;
    }
    ListNode getMid(ListNode head){
        if (head == null) return head;
        ListNode slow = head, fast = head;
        while(fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }

        return slow;
    }

    // No. 156
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null && root.right == null)
            return root;

        TreeNode newRoot = upsideDownBinaryTree(root.left);

        root.left.left = root.right;
        root.left.right = root;

        root.left = null;
        root.right = null;

        return newRoot;
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

    // No. 254
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> res = new ArrayList<>();
        dfsgetFactors(n, res, new ArrayList<>(), 2);
        return res;
    }
    public void dfsgetFactors(int n, List<List<Integer>> res, List<Integer> items, int start){
        if (n <= 1){
            if (items.size() > 1){
                res.add(new ArrayList<>(items));
            }
            return;
        }
        for (int i = start; i <= n; i++){
            if (n % i == 0){
                items.add(i);
                dfsgetFactors(n/i, res, items, i);
                items.remove(items.size() - 1);
            }
        }
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

    // No. 314
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        Map<TreeNode, Integer> colMap = new HashMap<>();
        queue.offer(root);
        colMap.put(root, 0);
        int min = 0, max = 0;
        while(!queue.isEmpty()){
            TreeNode cur = queue.poll();
            int col = colMap.get(cur);
            map.putIfAbsent(col, new ArrayList<>());
            map.get(col).add((Integer)cur.val);
            min = Math.min(min, col);
            max = Math.max(max, col);
            if (cur.left != null){
                queue.offer(cur.left);
                colMap.put(cur.left, col - 1);
            }
            if (cur.right != null){
                queue.offer(cur.right);
                colMap.put(cur.right, col + 1);
            }
        }
        for (int i = min; i <= max; i++){
            res.add(map.get(i));
        }
        return res;
    }

    // No. 329
    int[] dx = {0, 0, 1, -1};
    int[] dy = {1, -1, 0, 0};
    public int longestIncreasingPath(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return 0;
        int[][] cache = new int[matrix.length][matrix[0].length];
        int res = 0;
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[0].length; j++){
                res = Math.max(res, dplongestIncreasingPath(matrix, i, j, cache));
            }
        }
        return res;
    }
    private int dplongestIncreasingPath(int[][] matrix, int i, int j, int[][] cache){
        if(cache[i][j] > 0) return cache[i][j];
        cache[i][j] = 1;
        for (int k = 0; k < 4; k ++){
            int x = i + dx[k], y = j + dy[k];
            if (x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length && matrix[i][j] < matrix[x][y]){
                cache[i][j] = Math.max(cache[i][j], 1 + dplongestIncreasingPath(matrix, x, y, cache));
            }
        }
        return cache[i][j];
    }

    // No. 395
    public int longestSubstring(String s, int k) {
        return DClongestSubstring(s, 0, s.length(), k);
    }

    private int DClongestSubstring(String s, int start, int end, int k) {
        if (end < k) return 0;
        int[] countMap = new int[26];
        for (int i = start; i < end; i++)
            countMap[s.charAt(i) - 'a']++;
        for (int mid = start; mid < end; mid++) {
            if (countMap[s.charAt(mid) - 'a'] >= k) continue;
            int midNext = mid + 1;
            while (midNext < end && countMap[s.charAt(midNext) - 'a'] < k) midNext++;
            return Math.max(DClongestSubstring(s, start, mid, k),
                    DClongestSubstring(s, midNext, end, k));
        }
        return (end - start);
    }

    // No. 468
    public String validIPAddress(String queryIP) {
        if(queryIP.chars().filter(ch -> ch == '.').count() == 3){
            return validIPv4(queryIP);
        }
        if(queryIP.chars().filter(ch -> ch == ':').count() == 7){
            return validIPv6(queryIP);
        }
        return "Neither";
    }
    private String validIPv4(String IP){
        String[] nums = IP.split("\\.", -1);
        for(String x: nums){
            for(char ch: x.toCharArray()){
                if(!Character.isDigit(ch)) return "Neither";
            }
            if (x.length() == 0 || x.length() > 3){
                return "Neither";
            }
            if(x.charAt(0) == '0' && x.length() > 1) return "Neither";
            if(Integer.parseInt(x) > 255) return "Neither";
        }
        return "IPv4";
    }
    private String validIPv6(String IP){
        String[] nums = IP.split("\\:", -1);
        String hexdigits = "0123456789abcdefABCDEF";
        for(String x: nums){
            if(x.length() == 0 || x.length() > 4) return "Neither";
            for(char ch: x.toCharArray()){
                if(hexdigits.indexOf(ch) == -1) return "Neither";
            }
        }
        return "IPv6";
    }

    // No. 670
    public int maximumSwap(int num) {
        char[] digits = Integer.toString(num).toCharArray();
        int[] buckets = new int[10];
        for(int i = 0; i < digits.length; i++){
            buckets[digits[i] - '0'] = i;
        }
        for(int i = 0; i < digits.length; i++){
            for (int k = 9; k > digits[i] - '0'; k --){
                if (buckets[k] > i){
                    // Swap
                    char tmp = digits[i];
                    digits[i] = digits[buckets[k]];
                    digits[buckets[k]] = tmp;

                    return Integer.valueOf(new String(digits));
                }
            }
        }
        return num;
    }

    // No. 773
    private int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    public int slidingPuzzle(int[][] board) {
        int step = 0, m = 2, n = 3;
        HashSet<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        int state = getState(board);
        queue.add(state);
        visited.add(state);

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                state = queue.poll();
                if (state == 123450) return step;
                int pos = findZero(state);
                for (int[] d : dirs) {
                    int nr = pos / 3 + d[0];
                    int nc = pos % 3 + d[1];
                    if (nr < 0 || nr >= m || nc < 0 || nc >= n) continue;
                    int newState = move(state, pos, nr * 3 + nc);
                    if (visited.contains(newState)) continue;
                    queue.add(newState);
                    visited.add(newState);
                }
            }
            step++;
        }

        return -1;
    }
    private int getState(int[][] board) {
        int res = 0;
        for (int i = 0; i < 6; i++) {
            res = res * 10 + board[i / 3][i % 3];
        }
        return res;
    }
    private int findZero(int state) {
        for (int i = 0; i < 6; i++) {
            if (state % 10 == 0) {
                return 5 - i;
            }
            state /= 10;
        }
        return 0;
    }
    private int move(int state, int pos, int newPos) {
        int a = 0, s = state;
        for (int i = 0; i < 6; i++) {
            if (5 - i == newPos) a = s % 10;
            s /= 10;
        }
        int res = 0, base = 1;
        for (int i = 0; i < 6; i++) {
            if (5 - i == pos) res += a * base;
            else if (5 - i == newPos) ;
            else res += state % 10 * base;
            state /= 10;
            base *= 10;
        }
        return res;
    }

    // No. 992
    public int subarraysWithKDistinct(int[] nums, int k) {
        /*
        Time complexity: O(n)
        Space complexity: O(n)
         */
        int res = 0, prefix = 0;
        int [] m = new int[nums.length + 1];
        for (int i = 0, j = 0, cnt = 0; i < nums.length; i++){
            if (m[nums[i]] ++ == 0) ++ cnt;
            if (cnt > k){
                --m[nums[j++]];
                --cnt;
                prefix = 0;
            }
            while(m[nums[j]] > 1){
                ++ prefix;
                --m[nums[j++]];
            }
            if (cnt == k) res += prefix + 1;
        }
        return res;
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

    // No. 1143
    public int longestCommonSubsequence(String text1, String text2) {
        if(text1.length() == 0 || text2.length() == 0) return 0;
        int dp[][] = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 1; i < text1.length() + 1;i++){
            for (int j = 1; j < text2.length() + 1; j++){
                if(text1.charAt(i-1) == text2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else{
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[text1.length()][text2.length()];
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
