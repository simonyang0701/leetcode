import lib.*;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import lib.ListNode;


class SolutionTest{
    Solution solution = new Solution();
    @Test
    void rightSideView(){
        List<Integer> res = solution.rightSideView(new TreeNode<>(new Integer[]{1,2,3,null,5,null,4}));
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
        int res = solution.maxDepth(root);
        check(res, 3);
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