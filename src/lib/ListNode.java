package lib;

public class ListNode {
    public int val;
    public ListNode next;
    public ListNode() {}
    public ListNode(int[] array){
        ListNode root = null;
        for (int i = array.length - 1; i >= 0 ; i--)
            root = insert(root, array[i]);
        this.val = root.val;
        this.next = root.next;
    }
    public ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }

    static ListNode insert(ListNode root, int item)
    {
        ListNode temp = new ListNode();
        temp.val = item;
        temp.next = root;
        root = temp;
        return root;
    }

    static void print(ListNode root)
    {
        while (root != null)
        {
            System.out.print(root.val + " ");
            root = root.next;
        }
        System.out.println();
    }

    static boolean check(ListNode l1, ListNode l2){
        while(l1 != null && l2 != null && l1.val == l2.val) {
            l1 = l1.next;
            l2 = l2.next;
        }

        return l1 == l2;
    }
}
