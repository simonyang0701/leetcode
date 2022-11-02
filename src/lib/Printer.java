package lib;

import java.util.Arrays;

public class Printer {
    public static void print(Object o){
        if(o.getClass() == ListNode.class){
            ListNode.print((ListNode) o);
        }
        else if(o.getClass().isArray()){
            if (o.getClass().getComponentType().isArray()){
                System.out.println(Arrays.deepToString((Object[]) o));
            }else{
                System.out.println(Arrays.toString((int[]) o));
            }
        }
        else if(o.getClass() == TreeNode.class){
            BTreePrinter.printNode((TreeNode) o);
        }
        else {
            System.out.println(o);
        }
    }
}
