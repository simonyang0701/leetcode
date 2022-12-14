package lib;

import java.util.*;

public class Checker {
    public static void check(Object o1, Object o2){
        if(o1.getClass() != o2.getClass()){
            System.out.println(o1.getClass());
            System.out.println(o2.getClass());
            System.out.println("Datatype is mismatch!");
            assert false;
        }
        if(o1.getClass().isArray()){
            if(o1.getClass().getComponentType().isArray()){
                assert Arrays.deepEquals((Object[]) o1, (Object[]) o2);
            }else{
                assert Arrays.equals((int[]) o1, (int[]) o2);
            }
        }
        else if(o1.getClass() == ListNode.class){
            assert  ListNode.check((ListNode) o1, (ListNode) o2);
        }
        else if(o1.getClass() == ArrayList.class){
            if (((ArrayList<?>) o1).size() == 0 && ((ArrayList<?>) o2).size() == 0){
                assert true;
                return;
            }
            Iterator iter = ((ArrayList<?>) o1).iterator();
            Iterator iter2 = ((ArrayList<?>) o2).iterator();
            while (iter.hasNext()){
                assert iter.next().equals(iter2.next());
            }
            assert ((ArrayList<?>) o1).iterator().next().equals(((ArrayList<?>) o2).iterator().next());
        }
        else if(o1.getClass() == TreeNode.class){
            assert TreeNode.isSameTree((TreeNode) o1, (TreeNode) o2);
        }
        else{
            assert o1.equals(o2);
        }
    }
}
