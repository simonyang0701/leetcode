package lib;

import java.util.Arrays;

public class Printer {
    public static void print(Object o){
        if(o.getClass() == ListNode.class){
            ListNode.print((ListNode) o);
        }
        if(o.getClass().isArray()){
            System.out.println(Arrays.toString((int[]) o));
        }
        if(o.getClass() == TreeNode.class){
            BTreePrinter.printNode((TreeNode) o);
        }
    }
}
