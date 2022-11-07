import lib.*;
import org.junit.jupiter.api.Test;

import java.util.*;

import lib.ListNode;


class SolutionTest{
    Solution solution = new Solution();
    @Test
    void trap(){
        int res = solution.trap(new int[]{0,1,0,2,1,0,1,3,2,1,2,1});
        print(res);
        check(res, 6);
    }
    @Test
    void swapPairs(){
        ListNode res = solution.swapPairs(new ListNode(new int[]{1,2,3,4}));
        print(res);
        check(res, new ListNode(new int[]{2,1,4,3}));
    }
    @Test
    void kthSmallest(){
        int res = solution.kthSmallest(new TreeNode(new Integer[]{3,1,4,null,2}), 1);
        print(res);
        check(res, 1);
    }
    @Test
    void leastInterval(){
        int res = solution.leastInterval(new char[]{'A', 'A', 'A', 'B', 'B', 'B'}, 2);
        print(res);
        check(res, 8);
    }
    @Test
    void exist(){
        Boolean res = solution.exist(new char[][]{
                {'A', 'B', 'C','E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}
        }, "ABCCED");
        print(res);
        check(res, true);
    }
    @Test
    void buildTree(){
        TreeNode res = solution.buildTree(new int[]{3,9,20,15,7}, new int[]{9,3,15,20,7});
        print(res);
        check(res, new TreeNode(new Integer[]{3,9,20,null,null,15,7}));
    }
    @Test
    void lowestCommonAncestor(){
        TreeNode res = solution.lowestCommonAncestor(new TreeNode(new Integer[]{3,5,1,6,2,0,8,null,null,7,4}),
                new TreeNode(5), new TreeNode(1));
        print(res);
        check(res, new TreeNode(new Integer[]{3,5,1,6,2,0,8,null,null,7,4}));
    }
    @Test
    void insert(){
        int[][] res = solution.insert(new int[][]{{1,3}, {6,9}}, new int[]{2,5});
        print(res);
        check(res, new int[][]{{1,5}, {6,9}});
    }
    @Test
    void search(){
        int res = solution.search(new int[]{-1,0,3,5,9,12}, 9);
        print(res);
        check(res, 4);
    }
    @Test
    void isAnagram(){
        Boolean res = solution.isAnagram("anagram", "nagaram");
        print(res);
        check(res, true);
    }
    @Test
    void invertTree(){
        TreeNode res = solution.invertTree(new TreeNode(new Integer[]{4,2,7,1,3,6,9}));
        print(res);
        check(res, new TreeNode(new Integer[]{4,7,2,9,6,3,1}));
    }
    @Test
    void camelMatch(){
        List<Boolean> res = solution.camelMatch(new String[]{
                "FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"
        }, "FB");
        print(res);
        check(res, new ArrayList<>(Arrays.asList(new Boolean[]{true,false,true,true,false})));
    }
    @Test
    void isPalindrome(){
        Boolean res = solution.isPalindrome("A man, a plan, a canal: Panama");
        print(res);
        check(res, true);
    }
    @Test
    void maxProfit(){
        int res = solution.maxProfit(new int[]{7, 1, 5,3,6,4});
        print(res);
        check(res, 5);
    }
    @Test
    void mergeTwoLists(){
        ListNode res = solution.mergeTwoLists(new ListNode(new int[]{1,2,4}), new ListNode(new int[]{1,3,4}));
        print(res);
        check(res, new ListNode(new int[]{1,1,2,3,4,4}));
    }
    @Test
    void isValid(){
        Boolean res = solution.isValid("()");
        print(res);
        check(res, true);
    }
    @Test
    void longestCommonSubsequence(){
        int res = solution.longestCommonSubsequence("abcde", "ace");
        print(res);
        check(res, 3);
    }
    @Test
    void validIPAddress(){
        String res = solution.validIPAddress("172.16.254.1");
        print(res);
        check(res, "IPv4");
    }
    @Test
    void longestSubstring(){
        int res = solution.longestSubstring("aaabb", 3);
        print(res);
        check(res, 3);
    }
    @Test
    void getFactors(){
        List<List<Integer>> res = solution.getFactors(1);
        print(res);
        check(res, new ArrayList<>());
    }
    @Test
    void maxSubArray(){
        int res = solution.maxSubArray(new int[] {-2, 1, -3, 4, -1, 2, 1, -5, 4});
        print(res);
        check(res, 6);
    }
    @Test
    void jump(){
        int res = solution.jump(new int[]{2,3,1,1,4});
        print(res);
        check(res, 2);
    }
    @Test
    void maximumSwap(){
        int res = solution.maximumSwap(2736);
        print(res);
        check(res, 7236);
    }
    @Test
    void upsideDownBinaryTree(){
        TreeNode res = solution.upsideDownBinaryTree(new TreeNode(new Integer[]{1,2,3,4,5}));
        print(res);
        check(res, new TreeNode(new Integer[]{4,5,2,null,null,3,1}));
    }
    @Test
    void pathSum(){
        List<List<Integer>> res = solution.pathSum(new TreeNode(new Integer[]{5,4,8,11,null,13,4,7,2,null,null,5,1}), 22);
        print(res);
        List<List<Integer>> resCompare = new ArrayList<>();
        resCompare.add(Arrays.asList(5,4,11,2));
        resCompare.add(Arrays.asList(5,8,4,5));
        check(res, resCompare);
    }
    @Test
    void isValidBST(){
        Boolean res = solution.isValidBST(new TreeNode(new Integer[]{2, 1, 3}));
        print(res);
        check(res, true);
    }
    @Test
    void verticalOrder(){
        List<List<Integer>> res = solution.verticalOrder(new TreeNode(new Integer[]{3,9,8,4,0,1,7}));
        print(res);
        List<List<Integer>> resCompare = new ArrayList<>();
        resCompare.add(Arrays.asList(4));
        resCompare.add(Arrays.asList(9));
        resCompare.add(Arrays.asList(3, 0, 1));
        resCompare.add(Arrays.asList(8));
        resCompare.add(Arrays.asList(7));
        check(res, resCompare);
    }
    @Test
    void levelOrderBottom(){
        List<List<Integer>> res = solution.levelOrderBottom((new TreeNode(new Integer[]{3,9,20,null,null,15,7})));
        print(res);
        List<List<Integer>> resCompare = new ArrayList<>();
        resCompare.add(Arrays.asList(15, 7));
        resCompare.add(Arrays.asList(9, 20));
        resCompare.add(Arrays.asList(3));
        check(res, resCompare);
    }
    @Test
    void zigzagLevelOrder(){
        List<List<Integer>> res = solution.zigzagLevelOrder((new TreeNode(new Integer[]{3,9,20,null,null,15,7})));
        print(res);
        List<List<Integer>> resCompare = new ArrayList<>();
        resCompare.add(Arrays.asList(3));
        resCompare.add(Arrays.asList(20, 9));
        resCompare.add(Arrays.asList(15, 7));
        check(res, resCompare);
    }
    @Test
    void levelOrder(){
        List<List<Integer>> res = solution.levelOrder((new TreeNode(new Integer[]{3,9,20,null,null,15,7})));
        print(res);
        List<List<Integer>> resCompare = new ArrayList<>();
        resCompare.add(Arrays.asList(3));
        resCompare.add(Arrays.asList(9, 20));
        resCompare.add(Arrays.asList(15, 7));
        check(res, resCompare);
    }
    void sortList(){
        ListNode res = solution.sortList(new ListNode(new int[]{4,2,1,3}));
        print(res);
        check(res, new ListNode(new int[]{1,2,3,4}));
    }
    @Test
    void merge(){
        int[][] res = solution.merge(new int[][]{
                {1,3},{2,6},{8,10},{15,18}
        });
        print(res);
        check(res, new int[][]{{1,6},{8,10},{15,18}});
    }
    @Test
    void longestIncreasingPath(){
        int res = solution.longestIncreasingPath(new int[][]{{3,4,5},{3,2,6},{2,2,1}});
        print(res);
        check(res, 4);
    }
    @Test
    void subarraysWithKDistinct(){
        int res = solution.subarraysWithKDistinct(new int[]{1,2,1,2,3}, 2);
        print(res);
        check(res, 7);
    }
    @Test
    void letterCombinations(){
        List<String> res = solution.letterCombinations("23");
        print(res);
        check(res, new LinkedList<>(Arrays.asList("ad","ae","af","bd","be","bf","cd","ce","cf")));
    }
    @Test
    void threeSumClosest(){
        int res = solution.threeSumClosest(new int[]{-1, 2, 1, -4}, 1);
        print(res);
        check(res, 2);
    }
    @Test
    void threeSum(){
        List<List<Integer>> res = solution.threeSum(new int[]{-1,0,1,2,-1,-4});
        print(res);
        List<List<Integer>> resCompare = new ArrayList<>();
        resCompare.add(Arrays.asList(-1, -1, 2));
        resCompare.add(Arrays.asList(-1, 0, 1));
        check(res, resCompare);
    }
    @Test
    void intToRoman(){
        String res = solution.intToRoman(3);
        print(res);
        check(res, "III");
    }
    @Test
    void maxArea(){
        int res = solution.maxArea(new int[]{1,8,6,2,5,4,8,3,7});
        print(res);
        check(res, 49);
    }
    @Test
    void myAtoi(){
        int res = solution.myAtoi("42");
        print(res);
        check(res, 42);
    }
    @Test
    void slidingPuzzle(){
        int res = solution.slidingPuzzle(new int[][]{{1,2,3},{4,0,5}});
        print(res);
        check(res, 1);
    }
    @Test
    void rightSideView(){
        List<Integer> res = solution.rightSideView(new TreeNode(new Integer[]{1,2,3,null,5,null,4}));
        print(res);
        check(res, new ArrayList<>(Arrays.asList(1,3,4)));
    }
    @Test
    void subsets(){
        List<List<Integer>> res = solution.subsets(new int[] {1,2,3});
        print(res);
        check(res, twoDArrayToList(new Object[][]{
                {}, {1}, {1, 2}, {1, 2, 3}, {1, 3}, {2}, {2, 3}, {3}
        }));
    }
    @Test
    void multiply(){
        int[][] res = solution.multiply(new int[][]{
                {1,0,0},{-1,0,3}
        }, new int[][]{
                {7,0,0},{0,0,0},{0,0,1}
        });
        print(res);
        check(res, new int[][]{
                {7,0,0},{-7,0,3}
        });
    }
    @Test
    void searchRange(){
        int[] res = solution.searchRange(new int[]{5, 7, 7, 8, 8, 10}, 8);
        print(res);
        check(res, new int[]{3, 4});
    }
    @Test
    void minMutation(){
        int res = solution.minMutation("AACCGGTT", "AACCGGTA", new String[]{"AACCGGTA"});
        print(res);
        check(res, 1);
    }
    @Test
    void LRUCache(){
        Solution.LRUCache lRUCache = new Solution.LRUCache(2);
        lRUCache.put(1, 1); // cache is {1=1}
        lRUCache.put(2, 2); // cache is {1=1, 2=2}
        print(lRUCache.get(1));    // return 1
        lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
        print(lRUCache.get(2));    // returns -1 (not found)
        lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
        print(lRUCache.get(1));    // return -1 (not found)
        print(lRUCache.get(3));    // return 3
        print(lRUCache.get(4));    // return 4
    }
    @Test
    void minSubArrayLen(){
        int res = solution.minSubArrayLen(7, new int[]{2,3,1,2,4,3});
        print(res);
        check(res, 2);
    }
    @Test
    void TimeMap(){
        Solution.TimeMap timeMap = new Solution.TimeMap();
        timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
        print(timeMap.get("foo", 1));         // return "bar"
        print(timeMap.get("foo", 3));         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
        timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
        print(timeMap.get("foo", 4));         // return "bar2"
        print(timeMap.get("foo", 5));         // return "bar2"
    }
    @Test
    void validTree(){
        boolean res = solution.validTree(5, new int[][]{{0,1},{0,2},{0,3},{1,4}});
        print(res);
        check(res, true);
    }
    @Test
    void wordBreak(){
        boolean res = solution.wordBreak("leetcode", Arrays.asList("leet", "code"));
        print(res);
        check(res, true);

    }
    @Test
    void reverse(){
        int res = solution.reverse(123);
        print(res);
        check(res, 321);
    }
    @Test
    void convert(){
        String res = solution.convert("PAYPALISHIRING", 3);
        print(res);
        check(res, "PAHNAPLSIIGYIR");
    }
    @Test
    void longestPalindrome(){
        String res = solution.longestPalindrome("babad");
        print(res);
        check(res, "aba");
    }
    @Test
    void lengthOfLongestSubstring(){
        int res = solution.lengthOfLongestSubstring("abcabcbb");
        print(res);
        check(res, 3);
    }
    @Test
    void maxDepth(){
        TreeNode root = new TreeNode(new Integer[]{3,9,20,null,null,15,7});
        print(root);
//        int res = solution.maxDepth(root);
//        print(res);
//        check(res, 3);
    }
    @Test
    void twoSum(){
        int[] res = solution.twoSum(new int[]{2,7,11,15}, 9);
        print(res);
        check(res, new int[]{0, 1});
    }

    @Test
    void addTwoNumbers(){
        ListNode l1 = new ListNode(new int[]{2,4,3});
        ListNode l2 = new ListNode(new int[]{5,6,4});
        ListNode res = solution.addTwoNumbers(l1, l2);
        print(res);
        check(res, new ListNode(new int[]{7,0,8}));
    }
    void print(Object o){
        Printer.print(o);
    }
    void check(Object o1, Object o2){
        Checker.check(o1, o2);
    }

    public <T> List<List<T>> twoDArrayToList(T[][] twoDArray) {
        List<List<T>> list = new ArrayList<>();
        for (T[] array : twoDArray) {
            list.add(Arrays.asList(array));
        }
        return list;
    }
}