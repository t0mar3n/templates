use rayon::prelude::*;

use std::{ops::{Add, Sub, Mul}, iter::Sum};
type Matrix<T> = Vec<Vec<T>>;
type MatrixArray<T> = [Vec<T>];

pub trait CalcVecOps<T> {
    fn add(&self, other: &[T]) -> Vec<T>;
    fn sub(&self, other: &[T]) -> Vec<T>;
    fn hadamard(&self, other: &[T]) -> Vec<T>;
    fn scalar_mul(&self, scalar: T) -> Vec<T>;
    fn dot(&self, other: &[T]) -> T;
}

pub trait StdVecOps<T> {
    fn transpose(&self) -> Matrix<T>;
    fn apply_func<G: Sync + Send, F: Fn(T) -> G + Sync + Send>(&self, f: &F) -> Vec<G>;
    fn into_scalar(&self) -> Option<&T>;
    fn into_matrix(&self) -> Matrix<T>;
}

impl<T> CalcVecOps<T> for Vec<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn add(&self, other: &[T]) -> Vec<T> {
        assert_eq!(self.len(), other.len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s + *o)
            .collect()
    }

    fn sub(&self, other: &[T]) -> Vec<T> {
        assert_eq!(self.len(), other.len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s - *o)
            .collect()
    }

    fn hadamard(&self, other: &[T]) -> Vec<T> {
        assert_eq!(self.len(), other.len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s * *o)
            .collect()
    }

    fn scalar_mul(&self, scalar: T) -> Vec<T> {
        self.par_iter()
            .map(|v| *v * scalar)
            .collect()
    }

    fn dot(&self, other: &[T]) -> T {
        assert_eq!(self.len(), other.len());
        self.par_iter()
            .zip(other.par_iter())
            .map(|(s, o)| *s * *o)
            .sum::<T>()
    }
}

impl<T> StdVecOps<T> for Vec<T>
where
T: Copy + Default + Sync + Send,
{
    fn transpose(&self) -> Matrix<T> {
        self.par_iter()
            .map(|v| vec![*v])
            .collect()
    }

    fn apply_func<G: Sync + Send, F: Fn(T)->G + Sync + Send>(&self, f: &F) -> Vec<G> {
        self.par_iter()
            .map(|v| f(*v))
            .collect()
    }

    fn into_scalar(&self) -> Option<&T> {
        if self.len()==1 { self.get(0) }
        else { None }
    }

    fn into_matrix(&self) -> Matrix<T> {
        vec![self.to_vec()]
    }
}


impl<T> CalcVecOps<T> for &[T]
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn add(&self, other: &[T]) -> Vec<T> {
        assert_eq!(self.len(), other.len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s + *o)
            .collect()
    }

    fn sub(&self, other: &[T]) -> Vec<T> {
        assert_eq!(self.len(), other.len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s - *o)
            .collect()
    }

    fn hadamard(&self, other: &[T]) -> Vec<T> {
        assert_eq!(self.len(), other.len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s * *o)
            .collect()
    }

    fn scalar_mul(&self, scalar: T) -> Vec<T> {
        self.par_iter()
            .map(|v| *v * scalar)
            .collect()
    }

    fn dot(&self, other: &[T]) -> T {
        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)| *s * *o)
            .sum::<T>()
    }
}

impl<T> StdVecOps<T> for &[T]
where
T: Copy + Default + Sync + Send,
{
    fn transpose(&self) -> Matrix<T> {
        self.par_iter()
            .map(|v| vec![*v])
            .collect()
    }

    fn apply_func<G: Sync + Send, F: Fn(T)->G + Sync + Send>(&self, f: &F) -> Vec<G> {
        self.par_iter()
            .map(|v| f(*v))
            .collect()
    }

    fn into_scalar(&self) -> Option<&T> {
        if self.len()==1 { self.get(0) }
        else { None }
    }

    fn into_matrix(&self) -> Matrix<T> {
        vec![self.to_vec()]
    }
}


pub trait CalcMatrixOps<T> {
    fn add(&self, other: &MatrixArray<T>) -> Matrix<T>;
    fn sub(&self, other: &MatrixArray<T>) -> Matrix<T>;
    fn dot(&self, other: &MatrixArray<T>) -> Matrix<T>;
    fn scalar_mul(&self, scalar: T) -> Matrix<T>;
    fn hadamard(&self, other: &MatrixArray<T>) -> Matrix<T>;
}

pub trait StdMatrixOps<T> {
    fn transpose(&self) -> Matrix<T>;
    fn apply_func<G: Sync+Send, F: Fn(T) -> G + Sync + Send>(&self, f: &F) -> Matrix<G>;
    fn into_vec(&self) -> Option<&Vec<T>>;
}

impl<T> CalcMatrixOps<T> for Matrix<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn add(&self, other: &MatrixArray<T>) -> Matrix<T> {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.add(o)
            )
            .collect()
    }

    fn sub(&self, other: &MatrixArray<T>) -> Matrix<T> {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.sub(o)
            )
            .collect()
    }

    fn dot(&self, other: &[Vec<T>]) -> Matrix<T> {
        let m = self.len();    
        let n = self[0].len(); 
        let p = other[0].len();
    
        assert_eq!(n, other.len());

        (0..m).into_par_iter()
            .map(|i| 
                (0..p).into_par_iter()
                    .map(|j|
                        (0..n).into_par_iter()
                            .map(|k| self[i][k] * other[k][j])
                            .sum::<T>()
                    )
                    .collect()
            )
            .collect()
    }

    fn scalar_mul(&self, scalar: T) -> Matrix<T> {
        self.par_iter()
            .map(|ele|
                ele.scalar_mul(scalar)
            )
            .collect()
    }

    fn hadamard(&self, other: &MatrixArray<T>) -> Matrix<T> {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.hadamard(o)
            )
            .collect()
    }
}

impl<T> StdMatrixOps<T> for Matrix<T>
where
    T: Copy + Default + Sync + Send + Sum<T>
{
    fn transpose(&self) -> Matrix<T> {
        let n = self.len();
        let m = self[0].len();
        (0..m).into_par_iter()
            .map(|i|
                (0..n).into_par_iter()
                    .map(|j|
                        self[j][i]
                    )
                    .collect()
                )
            .collect()
    }

    fn apply_func<G: Sync + Send, F: Fn(T) -> G + Sync + Send>(&self, f: &F) -> Matrix<G> {
        self.par_iter()
            .map(|arr|
                arr.apply_func(f)
            )
            .collect()
    }

    fn into_vec(&self) -> Option<&Vec<T>> {
        if self.len()==1 { self.get(0) }
        else { None }
    }

}


impl<T> CalcMatrixOps<T> for &MatrixArray<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn add(&self, other: &MatrixArray<T>) -> Matrix<T> {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.add(o)
            )
            .collect()
    }

    fn sub(&self, other: &MatrixArray<T>) -> Matrix<T> {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.sub(o)
            )
            .collect()
    }

    fn dot(&self, other: &[Vec<T>]) -> Matrix<T> {
        let m = self.len();    
        let n = self[0].len(); 
        let p = other[0].len();
    
        assert_eq!(n, other.len());

        (0..m).into_par_iter()
            .map(|i| 
                (0..p).into_par_iter()
                    .map(|j|
                        (0..n).into_par_iter()
                            .map(|k| self[i][k] * other[k][j])
                            .sum::<T>()
                    )
                    .collect()
            )
            .collect()
    }

    fn scalar_mul(&self, scalar: T) -> Matrix<T> {
        self.par_iter()
            .map(|ele|
                ele.scalar_mul(scalar)
            )
            .collect()
    }

    fn hadamard(&self, other: &MatrixArray<T>) -> Matrix<T> {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.hadamard(o)
            )
            .collect()
    }
}

impl<T> StdMatrixOps<T> for &MatrixArray<T>
where
    T: Copy + Default + Sync + Send + Sum<T>
{
    fn transpose(&self) -> Matrix<T> {
        let n = self.len();
        let m = self[0].len();
        (0..m).into_par_iter()
            .map(|i|
                (0..n).into_par_iter()
                    .map(|j|
                        self[j][i]
                    )
                    .collect()
                )
            .collect()
    }

    fn apply_func<G: Sync + Send, F: Fn(T) -> G + Sync + Send>(&self, f: &F) -> Matrix<G> {
        self.par_iter()
            .map(|arr|
                arr.apply_func(f)
            )
            .collect()
    }

    fn into_vec(&self) -> Option<&Vec<T>> {
        if self.len()==1 { self.get(0) } 
        else { None }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = a.add(&b);
        assert_eq!(result, vec![5, 7, 9]);
    }

    #[test]
    fn test_sub() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = a.sub(&b);
        assert_eq!(result, vec![-3, -3, -3]);
    }
    
    #[test]
    fn test_hadamard() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = a.hadamard(&b);
        assert_eq!(result, vec![4, 10, 18]);
    }
    
    #[test]
    fn test_scalar_mul() {
        let a = vec![1, 2, 3];
        let scalar = 2;
        let result = a.scalar_mul(scalar);
        assert_eq!(result, vec![2, 4, 6]);
    }
    
    #[test]
    fn test_dot() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = a.dot(&b);
        assert_eq!(result, 32);
    }
    
    #[test]
    fn test_transpose() {
        let a = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let result = a.transpose();
        assert_eq!(result, vec![vec![1, 3, 5], vec![2, 4, 6]]);
    }

    #[test]
    fn test_matrix_add() {
        let a = vec![vec![1, 2], vec![3, 4]];
        let b = vec![vec![5, 6], vec![7, 8]];
        let result = a.add(&b);
        assert_eq!(result, vec![vec![6, 8], vec![10, 12]]);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let result = matrix.transpose();
        assert_eq!(result, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    }

    #[test]
    fn test_matrix_apply_func() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let result = matrix.apply_func(&|x| x * 2);
        assert_eq!(result, vec![vec![2, 4, 6], vec![8, 10, 12]]);
    }

    #[test]
    fn test_matrix_into_vec() {
        let matrix = vec![vec![1, 2, 3],];
        let result = matrix.into_vec();
        assert_eq!(result, Some(&vec![1, 2, 3]));
    }

    #[test]
    fn test_matrix_matrix_add() {
        let a = vec![vec![1, 2], vec![3, 4]];
        let b = vec![vec![5, 6], vec![7, 8]];
        let result = a.add(&b);
        assert_eq!(result, vec![vec![6, 8], vec![10, 12]]);
    }

    #[test]
    fn test_matrix_dot() {
        let a = vec![vec![1, 2], vec![3, 4]];
        let b = vec![vec![5, 6], vec![7, 8]];
        let result = a.dot(&b);
        assert_eq!(result, vec![vec![19, 22], vec![43, 50]]);
    }
}
