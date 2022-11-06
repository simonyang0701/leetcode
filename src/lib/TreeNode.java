package lib;

import java.util.LinkedList;
import java.util.Queue;

public class TreeNode {
    public TreeNode left, right;
    public int val;

    public TreeNode(int val) {
        this.val = val;
        this.right = null;
        this.left = null;
    }
    public TreeNode(Integer[] array){
        TreeNode root = new TreeNode(array[0]);
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        for (int i = 1; i < array.length; i++) {
            TreeNode node = q.peek();
            if (node.left == null) {
                if (array[i] != null){
                    node.left = new TreeNode(array[i]);
                    q.add(node.left);
                }
            } else if (node.right == null) {
                if (array[i] != null) {
                    node.right = new TreeNode(array[i]);
                    q.add(node.right);
                }
                q.remove();
            }
        }
        this.val = root.val;
        this.left = root.left;
        this.right = root.right;
    }
}
