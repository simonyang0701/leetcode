package lib;

import java.util.LinkedList;
import java.util.Queue;

public class TreeNode<T> {
    public TreeNode<T> left, right;
    public T val;

    public TreeNode(T val) {
        this.val = val;
        this.right = null;
        this.left = null;
    }
    public TreeNode(Integer[] array){
        if (array.length == 0) return;
        TreeNode root = new TreeNode(array[0]);
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        for (int i = 1; i < array.length; i++) {
            TreeNode node = q.peek();
            if (node.left == null || node.left.val == null) {
                node.left = new TreeNode(array[i]);
                if (array[i] != null) q.add(node.left);
            } else if (node.right == null || node.right.val == null) {
                node.right = new TreeNode(array[i]);
                if (array[i] != null) q.add(node.right);
                q.remove();
            }
        }
        this.val = (T) root.val;
        this.left = root.left;
        this.right = root.right;
    }
}
