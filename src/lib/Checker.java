package lib;

import java.util.Arrays;

public class Checker {
    public static void check(Object o1, Object o2){
        if(o1.getClass() != o2.getClass()){
            System.out.println("Datatype is mismatch!");
            assert false;
        }
        if(o1.getClass().isArray()){
            assert Arrays.equals((int[])o1, (int[])o2);
        }
        if(o1.getClass() == ListNode.class){
            assert  ListNode.check((ListNode) o1, (ListNode) o2);
        }
    }
}
