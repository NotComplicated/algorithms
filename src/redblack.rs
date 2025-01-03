use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::{self, FusedIterator, Peekable},
    marker::PhantomData,
    mem,
    ops::{BitAnd, BitOr, Index},
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
                parent,
                left,
                right,
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

    pub fn clear(&mut self) {
        drop_subtree(&mut self.root);
        self.len = 0;
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

    pub fn first(&self) -> Option<(&K, &V)> {
        leftmost(self.root).map(|node_ptr| {
            let node = unsafe { node_ptr.as_ref() };
            (&node.key, &node.value)
        })
    }

    pub fn first_mut(&mut self) -> Option<(&K, &mut V)> {
        leftmost(self.root).map(|mut node_ptr| {
            let node = unsafe { node_ptr.as_mut() };
            (&node.key, &mut node.value)
        })
    }

    pub fn last(&self) -> Option<(&K, &V)> {
        rightmost(self.root).map(|node_ptr| {
            let node = unsafe { node_ptr.as_ref() };
            (&node.key, &node.value)
        })
    }

    pub fn last_mut(&mut self) -> Option<(&K, &mut V)> {
        rightmost(self.root).map(|mut node_ptr| {
            let node = unsafe { node_ptr.as_mut() };
            (&node.key, &mut node.value)
        })
    }
}

impl<K, V> Default for Tree<K, V> {
    fn default() -> Self {
        Self::new()
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
    fn extend<IntoIter: IntoIterator<Item = (K, V)>>(&mut self, iter: IntoIter) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: Ord + Copy, V: Copy> Extend<(&'a K, &'a V)> for Tree<K, V> {
    fn extend<IntoIter: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: IntoIter) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for Tree<K, V> {
    fn from_iter<IntoIter: IntoIterator<Item = (K, V)>>(iter: IntoIter) -> Self {
        let mut tree = Self::new();
        tree.extend(iter);
        tree
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for Tree<K, V> {
    fn from(arr: [(K, V); N]) -> Self {
        Self::from_iter(arr)
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

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn min(mut self) -> Option<Self::Item> {
        self.next()
    }

    fn max(mut self) -> Option<Self::Item> {
        self.next_back()
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

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn min(mut self) -> Option<Self::Item> {
        self.next()
    }

    fn max(mut self) -> Option<Self::Item> {
        self.next_back()
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

pub type Map<K, V> = Tree<K, V>;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub struct Set<T>(Tree<T, ()>);

impl<T> Set<T> {
    pub fn new() -> Self {
        Self(Tree::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains<Q: ?Sized + Ord>(&self, item: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
    {
        self.0.contains_key(item)
    }

    pub fn is_subset(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        self.iter().all(|item| other.contains(item))
    }

    pub fn is_superset(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        other.is_subset(self)
    }

    pub fn insert(&mut self, item: T) -> bool
    where
        T: Ord,
    {
        self.0.insert(item, ()).is_none()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.into_iter()
    }

    pub fn first(&self) -> Option<&T> {
        self.0.first().map(|(key, _)| key)
    }

    pub fn last(&self) -> Option<&T> {
        self.0.last().map(|(key, _)| key)
    }

    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, T> {
        Union {
            prev: None,
            left: self.into_iter().peekable(),
            right: other.into_iter().peekable(),
        }
    }

    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, T> {
        Intersection {
            left: self.into_iter().peekable(),
            right: other.into_iter().peekable(),
        }
    }

    pub fn is_disjoint(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        self.union(other).next().is_none()
    }
}

impl<T> Default for Set<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord> Extend<T> for Set<T> {
    fn extend<IntoIter: IntoIterator<Item = T>>(&mut self, iter: IntoIter) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<'a, T: Ord + Copy> Extend<&'a T> for Set<T> {
    fn extend<IntoIter: IntoIterator<Item = &'a T>>(&mut self, iter: IntoIter) {
        self.extend(iter.into_iter().copied());
    }
}

impl<T: Ord> FromIterator<T> for Set<T> {
    fn from_iter<IntoIter: IntoIterator<Item = T>>(iter: IntoIter) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<T: Ord, const N: usize> From<[T; N]> for Set<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T: fmt::Debug> fmt::Debug for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<'set, T> IntoIterator for &'set Set<T> {
    type Item = &'set T;
    type IntoIter = iter::Map<Inorder<T, (), &'set ()>, fn((&'set T, &'set ())) -> &'set T>;

    fn into_iter(self) -> Self::IntoIter {
        Inorder::new(self.0.root).map(|(key, _)| key)
    }
}

pub struct Union<'a, T> {
    prev: Option<&'a T>,
    left: Peekable<<&'a Set<T> as IntoIterator>::IntoIter>,
    right: Peekable<<&'a Set<T> as IntoIterator>::IntoIter>,
}

impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = match (self.left.peek(), self.right.peek()) {
                (Some(l), Some(r)) if l <= r => self.left.next(),
                (Some(_), Some(_)) => self.right.next(),
                (Some(_), None) => self.left.next(),
                (None, Some(_)) => self.right.next(),
                (None, None) => None,
            };
            if next != self.prev {
                self.prev = next;
                break next;
            }
        }
    }
}

impl<'a, T: Ord> FusedIterator for Union<'a, T> {}

pub struct Intersection<'a, T> {
    left: Peekable<<&'a Set<T> as IntoIterator>::IntoIter>,
    right: Peekable<<&'a Set<T> as IntoIterator>::IntoIter>,
}

impl<'a, T: Ord> Iterator for Intersection<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.left.peek()?.cmp(self.right.peek()?) {
                Ordering::Less => self.left.next(),
                Ordering::Greater => self.right.next(),
                Ordering::Equal => break self.left.next(),
            };
        }
    }
}

impl<'a, T: Ord> FusedIterator for Intersection<'a, T> {}

impl<T: Clone + Ord> BitOr for &Set<T> {
    type Output = Set<T>;

    fn bitor(self, other: Self) -> Self::Output {
        self.union(other).cloned().collect()
    }
}

impl<T: Clone + Ord> BitAnd for &Set<T> {
    type Output = Set<T>;

    fn bitand(self, other: Self) -> Self::Output {
        self.intersection(other).cloned().collect()
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;

    #[test]
    fn empty() {
        let trees = [Tree::<(), ()>::new(), Default::default()];
        for tree in trees {
            assert_eq!(tree.len(), 0);
            assert!(tree.is_empty());
            assert!(tree.get(&()).is_none());
        }
    }

    #[test]
    fn insert() {
        let mut tree = Tree::new();
        assert_eq!(tree.insert(1, 1), None);
        assert_eq!(tree.insert(2, 2), None);
        assert_eq!(tree.insert(1, 3), Some(1));
        assert_eq!(tree.len(), 2);
        assert!(!tree.is_empty());
        let mut tree = Tree::new();
        assert_eq!(tree.insert("foo", 1), None);
        assert_eq!(tree.insert("bar", 2), None);
        assert_eq!(tree.insert("foo", 3), Some(1));
        assert_eq!(tree.len(), 2);
        assert!(!tree.is_empty());
        let mut tree = Tree::new();
        assert_eq!(tree.insert("foo".to_string(), 1), None);
        assert_eq!(tree.insert("bar".into(), 2), None);
        assert_eq!(tree.insert("foo".into(), 3), Some(1));
        assert_eq!(tree.len(), 2);
        assert!(!tree.is_empty());
    }

    #[test]
    fn get() {
        let mut tree = Tree::new();
        tree.insert(1, 1);
        tree.insert(2, 2);
        assert_eq!(tree.get(&1), Some(&1));
        assert_eq!(tree.get(&2), Some(&2));
        assert_eq!(tree.get(&3), None);
        let mut tree = Tree::new();
        tree.insert("foo", 1);
        tree.insert("bar", 2);
        assert_eq!(tree.get("foo"), Some(&1));
        assert_eq!(tree.get("bar"), Some(&2));
        assert_eq!(tree.get("baz"), None);
        let mut tree = Tree::new();
        tree.insert("foo".to_string(), 1);
        tree.insert("bar".into(), 2);
        assert_eq!(tree.get("foo"), Some(&1));
        assert_eq!(tree.get("bar"), Some(&2));
        assert_eq!(tree.get("baz"), None);
        let mut tree = Tree::new();
        tree.insert("foo".to_string(), vec![1, 2]);
        tree.insert("bar".into(), vec![3]);
        assert_eq!(tree.get("foo").map(|v| &**v), Some::<&[_]>(&[1, 2]));
        assert_eq!(tree.get("bar").map(|v| &**v), Some::<&[_]>(&[3]));
        assert_eq!(tree.get("baz"), None);
    }

    #[test]
    fn get_mut() {
        let mut tree = Tree::new();
        tree.insert("b".to_string(), "b".to_string());
        tree.get_mut("b").unwrap().push_str("az");
        assert_eq!(tree.get("b").map(|s| &**s), Some("baz"));
        tree.insert("a".to_string(), "a".to_string());
        tree.insert("c".to_string(), "c".to_string());
        tree.get_mut("c").unwrap().push_str("ar");
        assert_eq!(tree.get("c").map(|s| &**s), Some("car"));
        assert_eq!(format!("{tree:?}"), r#"{"a": "a", "b": "baz", "c": "car"}"#);
    }

    #[test]
    fn from_iter() {
        let tree = [("a", 1), ("b", 2), ("c", 3)]
            .into_iter()
            .collect::<Tree<_, _>>();
        assert_eq!(tree.len(), 3);
        assert_eq!(format!("{tree:?}"), r#"{"a": 1, "b": 2, "c": 3}"#);
    }

    #[test]
    fn extend() {
        let mut tree = Tree::new();
        tree.extend([("a", 1), ("c", 3)].iter().copied());
        assert_eq!(tree.len(), 2);
        assert_eq!(format!("{tree:?}"), r#"{"a": 1, "c": 3}"#);
        tree.extend([("d", 4), ("b", 2), ("e", 5)].iter().copied());
        assert_eq!(tree.len(), 5);
        assert_eq!(
            format!("{tree:?}"),
            r#"{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}"#
        );
    }

    #[test]
    fn clone() {
        let tree1 = [(2, 2), (3, 3), (1, 1)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree2 = tree1.clone();
        assert_eq!(tree1, tree2);
        assert_ne!(tree1.root, tree2.root);
    }

    #[test]
    fn clear() {
        let mut tree = Tree::new();
        tree.insert(1, 1);
        tree.insert(2, 2);
        tree.clear();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert!(tree.get(&1).is_none());
        assert!(tree.get(&2).is_none());
    }

    #[test]
    fn iter() {
        let mut tree = [(1, 1), (3, 3), (2, 2)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        assert_eq!(
            tree.iter().collect::<Vec<_>>(),
            vec![(&1, &1), (&2, &2), (&3, &3)]
        );
        assert_eq!(tree.keys().collect::<Vec<_>>(), vec![&1, &2, &3]);
        assert_eq!(tree.values().collect::<Vec<_>>(), vec![&1, &2, &3]);
        assert_eq!(tree.values_mut().collect::<Vec<_>>(), vec![&1, &2, &3]);
        let mut values = tree.values_mut();
        *values.next().unwrap() += 1;
        *values.next_back().unwrap() += 1;
        drop(values);
        assert_eq!(format!("{tree:?}"), r#"{1: 2, 2: 2, 3: 4}"#);
    }

    #[test]
    fn first_last() {
        let mut tree = Tree::new();
        assert_eq!(tree.first(), None);
        assert_eq!(tree.last(), None);
        tree.insert(1, 1);
        assert_eq!(tree.first(), Some((&1, &1)));
        assert_eq!(tree.last(), Some((&1, &1)));
        tree.insert(2, 2);
        assert_eq!(tree.first(), Some((&1, &1)));
        assert_eq!(tree.last(), Some((&2, &2)));
        tree.insert(3, 3);
        assert_eq!(tree.first(), Some((&1, &1)));
        assert_eq!(tree.last(), Some((&3, &3)));
        *tree.first_mut().unwrap().1 += 1;
        *tree.last_mut().unwrap().1 *= 2;
        assert_eq!(format!("{tree:?}"), r#"{1: 2, 2: 2, 3: 6}"#);
    }

    #[test]
    fn eq() {
        let tree1 = [(1, 1), (2, 2), (3, 3)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree2 = [(2, 2), (1, 1), (3, 3)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree3 = [(1, 1), (2, 2), (4, 4)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        assert_eq!(tree1, tree2);
        assert_ne!(tree1, tree3);
    }

    #[test]
    fn ord() {
        let tree1 = [('d', ()), ('b', ()), ('c', ())]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree2 = [('a', ()), ('b', ()), ('e', ())]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree3 = [('a', ()), ('a', ()), ('a', ())]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        assert!(tree1 > tree3);
        assert!(tree3 < tree1);
        assert!(tree1 > tree2);
        assert!(tree2 < tree1);
        assert!(tree2 > tree3);
        assert!(tree3 < tree2);
        let mut trees = [tree1.clone(), tree2.clone(), tree3.clone()];
        trees.sort();
        assert_eq!(trees, [tree3, tree2, tree1]);
    }

    #[test]
    fn index() {
        let tree = [(1, 1), (2, 2), (3, 3)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        assert_eq!(tree[&1], 1);
        assert_eq!(tree[&2], 2);
        assert_eq!(tree[&3], 3);
    }

    #[test]
    #[should_panic]
    fn index_panic() {
        let tree = Tree::<i32, ()>::new();
        let _ = tree[&1];
    }

    #[test]
    fn hash() {
        use std::collections::hash_map::DefaultHasher;

        let tree1 = [(1, 1), (2, 2), (3, 3)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree2 = [(2, 2), (1, 1), (3, 3)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let tree3 = [(1, 1), (2, 2), (4, 4)]
            .iter()
            .copied()
            .collect::<Tree<_, _>>();
        let mut hasher = DefaultHasher::new();
        tree1.hash(&mut hasher);
        let hash1 = hasher.finish();
        let mut hasher = DefaultHasher::new();
        tree2.hash(&mut hasher);
        let hash2 = hasher.finish();
        let mut hasher = DefaultHasher::new();
        tree3.hash(&mut hasher);
        let hash3 = hasher.finish();
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}

#[cfg(test)]
mod set_tests {
    use super::*;

    #[test]
    fn empty() {
        let sets = [Set::<()>::new(), Default::default()];
        for set in sets {
            assert_eq!(set.len(), 0);
            assert!(set.is_empty());
            assert!(!set.contains(&()));
        }
    }

    #[test]
    fn insert() {
        let mut set = Set::new();
        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(!set.insert(1));
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        let mut set = Set::new();
        assert!(set.insert("foo"));
        assert!(set.insert("bar"));
        assert!(!set.insert("foo"));
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        let mut set = Set::new();
        assert!(set.insert("foo".to_string()));
        assert!(set.insert("bar".into()));
        assert!(!set.insert("foo".into()));
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert_eq!(format!("{set:?}"), r#"{"bar", "foo"}"#);
    }

    #[test]
    fn contains() {
        let mut set = Set::new();
        set.insert("foo".to_string());
        set.insert("bar".into());
        assert!(set.contains("foo"));
        assert!(set.contains("bar"));
        assert!(!set.contains("baz"));
    }

    #[test]
    fn from_iter() {
        let set = ["b", "a", "a", "c"].iter().collect::<Set<_>>();
        assert_eq!(set.len(), 3);
        assert_eq!(format!("{set:?}"), r#"{"a", "b", "c"}"#);
    }

    #[test]
    fn union() {
        let set1 = ['a', 'b', 'c'].into_iter().collect::<Set<_>>();
        let set2 = ['b', 'c', 'd'].into_iter().collect::<Set<_>>();
        assert!(set1.union(&set2).copied().eq(['a', 'b', 'c', 'd']));
        let union = &set1 | &set2;
        assert_eq!(format!("{union:?}"), r#"{'a', 'b', 'c', 'd'}"#);
    }

    #[test]
    fn intersection() {
        let set1 = ['a', 'b', 'c'].into_iter().collect::<Set<_>>();
        let set2 = ['b', 'c', 'd'].into_iter().collect::<Set<_>>();
        assert!(set1.intersection(&set2).copied().eq(['b', 'c']));
        let intersection = &set1 & &set2;
        assert_eq!(format!("{intersection:?}"), r#"{'b', 'c'}"#);
    }
}
