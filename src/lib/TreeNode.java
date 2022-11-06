package lib;

import java.util.LinkedList;
import java.util.Queue;

public class TreeNode {
    public TreeNode left, right;
    public Integer val;
    public TreeNode(Integer val) {
        this.val = val;
    }
    public TreeNode(Integer[] array){
        TreeNode root = new TreeNode(array[0]);
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        for (int i = 1; i < array.length; i++) {
            TreeNode node = q.peek();
            if (node.left == null) {
                node.left = new TreeNode(array[i]);
                if (array[i] != null) q.add(node.left);
            } else if (node.right == null) {
                node.right = new TreeNode(array[i]);
                if (array[i] != null) q.add(node.right);
                q.remove();
            }
        }
        deleteNullTreeNode(root);
        this.val = root.val;
        this.left = root.left;
        this.right = root.right;
    }
    private void deleteNullTreeNode(TreeNode node){
        if(node == null) return;
        try{
            int left = node.left.val;
        }catch (Exception e){
            node.left = null;
        }
        try{
            int right = node.right.val;
        }catch (Exception e){
            node.right = null;
        }
        deleteNullTreeNode(node.left);
        deleteNullTreeNode(node.right);
    }
    public static boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val == q.val){
            boolean left = isSameTree(p.left, q.left);
            boolean right = isSameTree(p.right, q.right);
            return (left & right);
        }
        return false;
    }
}
