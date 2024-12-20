use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    ops::Index,
    ptr::NonNull,
};

#[derive(Copy, Clone, PartialEq)]
enum Color {
    Red,
    Black,
}

type NodeRef<K, V> = Option<NonNull<Node<K, V>>>;

struct Node<K, V> {
    key: K,
    value: V,
    color: Color,
    parent: NodeRef<K, V>,
    left: NodeRef<K, V>,
    right: NodeRef<K, V>,
}

impl<K, V> Node<K, V> {
    fn new(
        key: K,
        value: V,
        color: Color,
        parent: NodeRef<K, V>,
        left: NodeRef<K, V>,
        right: NodeRef<K, V>,
    ) -> NodeRef<K, V> {
        Some(unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                key,
                value,
                color,
                parent: None,
                left: None,
                right: None,
            })))
        })
    }
}

fn leftmost<K, V>(mut node_ref: NodeRef<K, V>) -> NodeRef<K, V> {
    let mut left = node_ref;
    while let Some(node_ptr) = node_ref {
        left = node_ref;
        node_ref = unsafe { node_ptr.as_ref() }.left;
    }
    left
}

fn rightmost<K, V>(mut node_ref: NodeRef<K, V>) -> NodeRef<K, V> {
    let mut right = node_ref;
    while let Some(node_ptr) = node_ref {
        right = node_ref;
        node_ref = unsafe { node_ptr.as_ref() }.right;
    }
    right
}

fn rotate_left<K, V>(root_ref: &mut NodeRef<K, V>, mut node_ptr: NonNull<Node<K, V>>) {
    let node = unsafe { node_ptr.as_mut() };
    if let Some(mut right_ptr) = node.right {
        let right = unsafe { right_ptr.as_mut() };
        let right_left_ref = right.left;
        node.right = right_left_ref;
        if let Some(mut right_left_ptr) = right_left_ref {
            unsafe { right_left_ptr.as_mut() }.parent = Some(node_ptr);
        }
        right.parent = node.parent;
        if let Some(mut parent_ptr) = node.parent {
            let parent = unsafe { parent_ptr.as_mut() };
            if parent.left == Some(node_ptr) {
                parent.left = Some(right_ptr);
            } else {
                parent.right = Some(right_ptr);
            }
        } else {
            *root_ref = Some(right_ptr);
        }
        right.left = Some(node_ptr);
        node.parent = Some(right_ptr);
    }
}

fn rotate_right<K, V>(root_ref: &mut NodeRef<K, V>, mut node_ptr: NonNull<Node<K, V>>) {
    let node = unsafe { node_ptr.as_mut() };
    if let Some(mut left_ptr) = node.left {
        let left = unsafe { left_ptr.as_mut() };
        let left_right_ref = left.right;
        node.left = left_right_ref;
        if let Some(mut left_right_ptr) = left_right_ref {
            unsafe { left_right_ptr.as_mut() }.parent = Some(node_ptr);
        }
        left.parent = node.parent;
        if let Some(mut parent_ptr) = node.parent {
            let parent = unsafe { parent_ptr.as_mut() };
            if parent.left == Some(node_ptr) {
                parent.left = Some(left_ptr);
            } else {
                parent.right = Some(left_ptr);
            }
        } else {
            *root_ref = Some(left_ptr);
        }
        left.right = Some(node_ptr);
        node.parent = Some(left_ptr);
    }
}

fn fixup<K, V>(root_ref: &mut NodeRef<K, V>, mut node_ptr: NonNull<Node<K, V>>) {
    loop {
        let node = unsafe { node_ptr.as_mut() };
        let Some(mut parent_ptr) = node.parent else {
            break;
        };
        let parent = unsafe { parent_ptr.as_mut() };
        if parent.color == Color::Black {
            break;
        }
        let grandparent = unsafe { parent.parent.unwrap_unchecked().as_mut() };
        if node.parent == grandparent.left {
            let uncle_ref = grandparent.right;
            if uncle_ref.is_some_and(|uncle_ptr| unsafe { uncle_ptr.as_ref().color } == Color::Red)
            {
                let uncle = unsafe { uncle_ref.unwrap_unchecked().as_mut() };
                parent.color = Color::Black;
                uncle.color = Color::Black;
                grandparent.color = Color::Red;
                node_ptr = unsafe { parent.parent.unwrap_unchecked() };
            } else {
                if parent.right == Some(node_ptr) {
                    node_ptr = parent_ptr;
                    drop((node, parent, grandparent));
                    rotate_left(root_ref, node_ptr);
                }
                let node = unsafe { node_ptr.as_mut() };
                let parent = unsafe { node.parent.unwrap_unchecked().as_mut() };
                let mut grandparent_ptr = unsafe { parent.parent.unwrap_unchecked() };
                let grandparent = unsafe { grandparent_ptr.as_mut() };
                parent.color = Color::Black;
                grandparent.color = Color::Red;
                drop((node, parent, grandparent));
                rotate_right(root_ref, grandparent_ptr);
            }
        } else {
            let uncle_ref = grandparent.left;
            if uncle_ref.is_some_and(|uncle_ptr| unsafe { uncle_ptr.as_ref().color } == Color::Red)
            {
                let uncle = unsafe { uncle_ref.unwrap_unchecked().as_mut() };
                parent.color = Color::Black;
                uncle.color = Color::Black;
                grandparent.color = Color::Red;
                node_ptr = unsafe { parent.parent.unwrap_unchecked() };
            } else {
                if parent.left == Some(node_ptr) {
                    node_ptr = parent_ptr;
                    drop((node, parent, grandparent));
                    rotate_right(root_ref, node_ptr);
                }
                let node = unsafe { node_ptr.as_mut() };
                let parent = unsafe { node.parent.unwrap_unchecked().as_mut() };
                let mut grandparent_ptr = unsafe { parent.parent.unwrap_unchecked() };
                let grandparent = unsafe { grandparent_ptr.as_mut() };
                parent.color = Color::Black;
                grandparent.color = Color::Red;
                drop((node, parent, grandparent));
                rotate_left(root_ref, grandparent_ptr);
            }
        }
    }
    if let Some(mut root_ptr) = *root_ref {
        unsafe { root_ptr.as_mut() }.color = Color::Black;
    }
}

fn clone_subtree<K: Clone, V: Clone>(
    node_ref: NodeRef<K, V>,
    parent: NodeRef<K, V>,
) -> NodeRef<K, V> {
    node_ref.and_then(|node_ptr| {
        let node = unsafe { node_ptr.as_ref() };
        let clone = Node::new(
            node.key.clone(),
            node.value.clone(),
            node.color,
            parent,
            None,
            None,
        );
        unsafe {
            let clone_ptr = clone.unwrap_unchecked().as_ptr();
            (*clone_ptr).left = clone_subtree(node.left, clone);
            (*clone_ptr).right = clone_subtree(node.right, clone);
        }
        clone
    })
}

fn drop_subtree<K, V>(node_ref: &mut NodeRef<K, V>) {
    if let Some(node_ptr) = node_ref.take() {
        let mut node = unsafe { Box::from_raw(node_ptr.as_ptr()) };
        drop_subtree(&mut node.left);
        drop_subtree(&mut node.right);
    }
}

#[derive(Default)]
pub struct Tree<K, V> {
    root: NodeRef<K, V>,
    len: usize,
}

impl<K, V> Tree<K, V> {
    pub fn new() -> Self {
        Tree { root: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn get<Q: ?Sized + Ord>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
    {
        let mut node_ref = self.root;
        loop {
            let node = unsafe { node_ref?.as_ref() };
            node_ref = match key.borrow().cmp(&node.key.borrow()) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => break Some(&node.value),
            }
        }
    }

    pub fn get_mut<Q: ?Sized + Ord>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
    {
        let mut node_ref = self.root;
        loop {
            let node = unsafe { node_ref?.as_mut() };
            node_ref = match key.borrow().cmp(&node.key.borrow()) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => break Some(&mut node.value),
            }
        }
    }

    pub fn contains_key<Q: ?Sized + Ord>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
    {
        self.get(key).is_some()
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        let mut parent_ref = None;
        let mut node_ref = self.root;
        while let Some(mut node_ptr) = node_ref {
            parent_ref = node_ref;
            let node = unsafe { node_ptr.as_mut() };
            node_ref = match key.cmp(&node.key) {
                Ordering::Less => node.left,
                Ordering::Greater => node.right,
                Ordering::Equal => return Some(mem::replace(&mut node.value, value)),
            };
        }
        if let Some(mut parent_ptr) = parent_ref {
            let parent = unsafe { parent_ptr.as_mut() };
            let child = if key < parent.key {
                &mut parent.left
            } else {
                &mut parent.right
            };
            *child = Node::new(key, value, Color::Red, parent_ref, None, None);
            fixup(&mut self.root, unsafe { child.unwrap_unchecked() });
        } else {
            self.root = Node::new(key, value, Color::Black, None, None, None);
        }
        self.len += 1;
        None
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.into_iter()
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> {
        self.into_iter().map(|(key, _)| key)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.into_iter().map(|(_, value)| value)
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut V> {
        self.into_iter().map(|(_, value)| value)
    }

    pub fn min(&self) -> Option<(&K, &V)> {
        leftmost(self.root).map(|node_ptr| {
            let node = unsafe { node_ptr.as_ref() };
            (&node.key, &node.value)
        })
    }

    pub fn max(&self) -> Option<(&K, &V)> {
        rightmost(self.root).map(|node_ptr| {
            let node = unsafe { node_ptr.as_ref() };
            (&node.key, &node.value)
        })
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for Tree<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other)
    }
}

impl<K: Eq, V: Eq> Eq for Tree<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for Tree<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<K: Ord, V: Ord> Ord for Tree<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<K: Ord, V> Extend<(K, V)> for Tree<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: Ord + Copy, V: Copy> Extend<(&'a K, &'a V)> for Tree<K, V> {
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K: Clone, V: Clone> Clone for Tree<K, V> {
    fn clone(&self) -> Self {
        Self {
            root: clone_subtree(self.root, None),
            len: self.len,
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Tree<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<Q: ?Sized + Ord, K: Borrow<Q> + Ord, V> Index<&Q> for Tree<K, V> {
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K: Hash, V: Hash> Hash for Tree<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (key, value) in self {
            key.hash(state);
            value.hash(state);
        }
    }
}

impl<'tree, K, V> IntoIterator for &'tree Tree<K, V> {
    type Item = (&'tree K, &'tree V);
    type IntoIter = Inorder<K, V, &'tree ()>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self.root)
    }
}

impl<'tree, K, V> IntoIterator for &'tree mut Tree<K, V> {
    type Item = (&'tree K, &'tree mut V);
    type IntoIter = Inorder<K, V, &'tree mut ()>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self.root)
    }
}

impl<K, V> Drop for Tree<K, V> {
    fn drop(&mut self) {
        drop_subtree(&mut self.root);
    }
}

pub struct Inorder<K, V, Ref> {
    left: NodeRef<K, V>,
    right: NodeRef<K, V>,
    phantom: PhantomData<Ref>,
}

impl<K, V, Ref> Inorder<K, V, Ref> {
    fn new(root: NodeRef<K, V>) -> Self {
        Self {
            left: leftmost(root),
            right: rightmost(root),
            phantom: PhantomData,
        }
    }
}

impl<K, V, Ref> FusedIterator for Inorder<K, V, Ref> where Self: Iterator {}

impl<'tree, K: 'tree, V: 'tree> Iterator for Inorder<K, V, &'tree ()> {
    type Item = (&'tree K, &'tree V);

    fn next(&mut self) -> Option<Self::Item> {
        let node_ptr = self.left?;
        let node = unsafe { node_ptr.as_ref() };
        if node_ptr == unsafe { self.right.unwrap_unchecked() } {
            (self.left, self.right) = (None, None);
        } else if node.right.is_some() {
            self.left = leftmost(node.right);
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
        Some((&node.key, &node.value))
    }
}

impl<'tree, K: 'tree, V: 'tree> DoubleEndedIterator for Inorder<K, V, &'tree ()> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let node_ptr = self.right?;
        let node = unsafe { node_ptr.as_ref() };
        if node_ptr == unsafe { self.left.unwrap_unchecked() } {
            (self.left, self.right) = (None, None);
        } else if node.left.is_some() {
            self.right = rightmost(node.left);
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
        Some((&node.key, &node.value))
    }
}

impl<'tree, K: 'tree, V: 'tree> Iterator for Inorder<K, V, &'tree mut ()> {
    type Item = (&'tree K, &'tree mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let mut node_ptr = self.left?;
        let node = unsafe { node_ptr.as_mut() };
        if node_ptr == unsafe { self.right.unwrap_unchecked() } {
            (self.left, self.right) = (None, None);
        } else if node.right.is_some() {
            self.left = leftmost(node.right);
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
        Some((&node.key, &mut node.value))
    }
}

impl<'tree, K: 'tree, V: 'tree> DoubleEndedIterator for Inorder<K, V, &'tree mut ()> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut node_ptr = self.right?;
        let node = unsafe { node_ptr.as_mut() };
        if node_ptr == unsafe { self.left.unwrap_unchecked() } {
            (self.left, self.right) = (None, None);
        } else if node.left.is_some() {
            self.right = rightmost(node.left);
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
        Some((&node.key, &mut node.value))
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Default, Clone, Hash)]
pub struct Set<T>(Tree<T, ()>);

impl<T> Set<T> {
    pub fn new() -> Self {
        Self(Tree::new())
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.0.into_iter().map(|(key, _)| key)
    }

    pub fn contains<Q: Ord>(&self, item: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
    {
        self.0.contains_key(item)
    }

    pub fn min(&self) -> Option<&T> {
        self.0.min().map(|(key, _)| key)
    }

    pub fn max(&self) -> Option<&T> {
        self.0.max().map(|(key, _)| key)
    }
}

impl<T: fmt::Debug> fmt::Debug for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
