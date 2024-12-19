use std::{borrow::Borrow, cmp::Ordering, marker::PhantomData, ptr::NonNull};

#[derive(Copy, Clone, PartialEq)]
enum Color {
    Red,
    Black,
}

type NodeRef<K, V> = Option<NonNull<Node<K, V>>>;

struct Node<K, V> {
    key: K,
    val: V,
    color: Color,
    parent: NodeRef<K, V>,
    left: NodeRef<K, V>,
    right: NodeRef<K, V>,
}

impl<K, V> Node<K, V> {
    fn new(
        key: K,
        val: V,
        color: Color,
        parent: NodeRef<K, V>,
        left: NodeRef<K, V>,
        right: NodeRef<K, V>,
    ) -> NodeRef<K, V> {
        Some(unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                key,
                val,
                color,
                parent: None,
                left: None,
                right: None,
            })))
        })
    }
}

fn color<K, V>(node_ref: NodeRef<K, V>) -> Color {
    node_ref.map_or(Color::Black, |node_ptr| unsafe { node_ptr.as_ref() }.color)
}

fn drop_subtree<K, V>(node_ref: &mut NodeRef<K, V>) {
    if let Some(node_ptr) = node_ref.take() {
        let mut node = unsafe { Box::from_raw(node_ptr.as_ptr()) };
        drop_subtree(&mut node.left);
        drop_subtree(&mut node.right);
    }
}

pub struct Tree<K, V> {
    root: NodeRef<K, V>,
}

impl<K, V> Tree<K, V> {
    pub fn new() -> Self {
        Tree { root: None }
    }

    pub fn get<Q: Ord>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        let mut node_ref = self.root;
        loop {
            let node = unsafe { node_ref?.as_ref() };
            node_ref = match key.borrow().cmp(&node.key.borrow()) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => break Some(&node.val),
            }
        }
    }

    pub fn get_mut<Q: Ord>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        let mut node_ref = self.root;
        loop {
            let node = unsafe { node_ref?.as_mut() };
            node_ref = match key.borrow().cmp(&node.key.borrow()) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => break Some(&mut node.val),
            }
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.into_iter()
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> {
        self.into_iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.into_iter().map(|(_, v)| v)
    }
}

impl<'tree, K, V> IntoIterator for &'tree Tree<K, V> {
    type Item = (&'tree K, &'tree V);
    type IntoIter = Inorder<'tree, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut node_ref = self.root;
        let mut left = node_ref;
        while let Some(node_ptr) = node_ref {
            left = node_ref;
            node_ref = unsafe { node_ptr.as_ref() }.left;
        }
        node_ref = self.root;
        let mut right = node_ref;
        while let Some(node_ptr) = node_ref {
            right = node_ref;
            node_ref = unsafe { node_ptr.as_ref() }.right;
        }
        Self::IntoIter {
            left,
            right,
            _ref: PhantomData,
        }
    }
}

impl<K, V> Drop for Tree<K, V> {
    fn drop(&mut self) {
        drop_subtree(&mut self.root);
    }
}

pub struct Inorder<'tree, K, V> {
    left: NodeRef<K, V>,
    right: NodeRef<K, V>,
    _ref: PhantomData<&'tree ()>,
}

impl<'tree, K: 'tree, V: 'tree> Iterator for Inorder<'tree, K, V> {
    type Item = (&'tree K, &'tree V);

    fn next(&mut self) -> Option<Self::Item> {
        let node_ptr = self.left?;
        let node = unsafe { node_ptr.as_ref() };
        if node_ptr == unsafe { self.right.unwrap_unchecked() } {
            (self.left, self.right) = (None, None);
        } else if node.right.is_some() {
            let mut node_ref = node.right;
            self.left = node_ref;
            while let Some(node_ptr) = node_ref {
                self.left = node_ref;
                node_ref = unsafe { node_ptr.as_ref() }.left;
            }
        } else {
            loop {
                let node_ref = self.left;
                self.left = unsafe { self.left.unwrap_unchecked().as_ref() }.parent;
                let left = unsafe { self.left.unwrap_unchecked().as_ref() }.left;
                if left == node_ref {
                    break;
                }
            }
        }
        Some((&node.key, &node.val))
    }
}

impl<'tree, K: 'tree, V: 'tree> DoubleEndedIterator for Inorder<'tree, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let node_ptr = self.right?;
        let node = unsafe { node_ptr.as_ref() };
        if node_ptr == unsafe { self.left.unwrap_unchecked() } {
            (self.left, self.right) = (None, None);
        } else if node.left.is_some() {
            let mut node_ref = node.left;
            self.right = node_ref;
            while let Some(node_ptr) = node_ref {
                self.right = node_ref;
                node_ref = unsafe { node_ptr.as_ref() }.right;
            }
        } else {
            loop {
                let node_ref = self.right;
                self.right = unsafe { self.right.unwrap_unchecked().as_ref() }.parent;
                let right = unsafe { self.right.unwrap_unchecked().as_ref() }.right;
                if right == node_ref {
                    break;
                }
            }
        }
        Some((&node.key, &node.val))
    }
}
