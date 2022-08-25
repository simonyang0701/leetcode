import lib.Checker;
import lib.ListNode;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Objects;

import lib.Printer;
import lib.ListNode;


class SolutionTest{
    Solution solution = new Solution();
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
}