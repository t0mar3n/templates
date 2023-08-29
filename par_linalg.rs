use rayon::prelude::*;

use std::{ops::{Add, Sub, Mul}, iter::Sum};
type Matrix<T> = Vec<Vec<T>>;
type MatrixArray<T> = [Vec<T>];

pub trait CalcVecOps<T> {
    fn add(&self, other: &[T]) -> Vec<T>;
    fn sub(&self, other: &[T]) -> Vec<T>;
    fn hadamard(&self, other: &[T]) -> Vec<T>;
    fn scalar_mul(&self, scalar: T) -> Vec<T>;
    fn dot(&self, other: &[T]) -> Option<T>;
}

pub trait StdVecOps<T> {
    fn transpose(&self) -> Matrix<T>;
    fn apply_func<G: Sync + Send, F: Fn(T) -> G + Sync + Send>(&self, f: &F) -> Vec<G>;
    fn into_scalar(&self) -> Option<&T>;
    fn into_matrix(&self) -> Matrix<T>;
}

pub trait VecOpsf64<T> {
    fn norm2(&self) -> f64;
    fn normalized(&self) -> Vec<f64>;
    fn cos_similarity(&self, other: &[T]) -> f64;
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

    fn dot(&self, other: &[T]) -> Option<T> {
        self.vec_mtx_mul(&other.transpose()).into_scalar().map(|f| *f)
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
        self.get(0)
    }

    fn into_matrix(&self) -> Matrix<T> {
        vec![self.iter().map(|&v| v).collect::<Vec<_>>()]
    }
}


impl<T> VecOpsf64<T> for Vec<T>
where
    T: Copy + Default + Into<f64> + From<f64> + Sync + Send,
{
    fn norm2(&self) -> f64 {
        self.par_iter()
            .map(|v| {let v = (*v).into(); v*v})
            .sum()
    }

    fn normalized(&self) -> Vec<f64> {
        let norm_sq = self.norm2().sqrt();
        self.apply_func(&|x: T|->f64{x.into()/norm_sq})
    }

    fn cos_similarity(&self, other: &[T]) -> f64 {
        let s = self.normalized();
        let o = other.normalized();
        s.dot(&o).unwrap()
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

    fn dot(&self, other: &[T]) -> Option<T> {
        self.vec_mtx_mul(&other.transpose()).into_scalar().map(|f| *f)
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
        self.get(0)
    }

    fn into_matrix(&self) -> Matrix<T> {
        vec![self.iter().map(|&v| v).collect::<Vec<_>>()]
    }
}


impl<T> VecOpsf64<T> for &[T]
where
    T: Copy + Default + Into<f64> + From<f64> + Sync + Send,
{
    fn norm2(&self) -> f64 {
        self.par_iter()
            .map(|v| {let v = (*v).into(); v*v})
            .sum()
    }

    fn normalized(&self) -> Vec<f64> {
        let norm_sq = self.norm2().sqrt();
        self.apply_func(&|x: T|->f64{x.into()/norm_sq})
    }

    fn cos_similarity(&self, other: &[T]) -> f64 {
        let s = self.normalized();
        let o = other.normalized();
        s.dot(&o).unwrap()
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
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.add(o)
            )
            .collect()
    }

    fn sub(&self, other: &MatrixArray<T>) -> Matrix<T> {
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.sub(o)
            )
            .collect()
    }

    fn dot(&self, other: &MatrixArray<T>) -> Matrix<T> {
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        let other_trans = other.transpose();

        self
            .par_iter()
            .map(|s|
                other_trans
                    .par_iter()
                    .map(|o|
                        s.hadamard(o).into_par_iter().sum::<T>()
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
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.par_iter()
                    .zip(o.par_iter())
                    .map(|(sv, ov)| *sv * *ov)
                    .collect()
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
        self.get(0)
    }

}


impl<T> CalcMatrixOps<T> for &MatrixArray<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn add(&self, other: &MatrixArray<T>) -> Matrix<T> {
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.add(o)
            )
            .collect()
    }

    fn sub(&self, other: &MatrixArray<T>) -> Matrix<T> {
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.sub(o)
            )
            .collect()
    }

    fn dot(&self, other: &MatrixArray<T>) -> Matrix<T> {
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        let other_trans = other.transpose();

        self
            .par_iter()
            .map(|s|
                other_trans
                    .par_iter()
                    .map(|o|
                        s.hadamard(o).into_par_iter().sum::<T>()
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
        let m = self[0].len();
        let n = other.len();
        assert_eq!(m, n);

        self.par_iter()
            .zip(other.par_iter())
            .map(|(s,o)|
                s.par_iter()
                    .zip(o.par_iter())
                    .map(|(sv, ov)| *sv * *ov)
                    .collect()
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
        self.get(0)
    }

}

pub trait VectorMatrixOps<T> {
    fn vec_mtx_mul(&self, matrix: &MatrixArray<T>) -> Vec<T>;
}

impl<T> VectorMatrixOps<T> for Vec<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn vec_mtx_mul(&self, matrix: &MatrixArray<T>) -> Vec<T> {
        let mat_trans = matrix.transpose();
        mat_trans.par_iter()
            .map(|col| self.hadamard(col).into_par_iter().sum::<T>())
            .collect()
    }

}

impl<T> VectorMatrixOps<T> for &[T]
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + Default + Sync + Send + Sum<T>
{
    fn vec_mtx_mul(&self, matrix: &MatrixArray<T>) -> Vec<T> {
        let mat_trans = matrix.transpose();
        mat_trans.par_iter()
            .map(|col| self.hadamard(col).into_par_iter().sum::<T>())
            .collect()
    }

}

#[test]
fn test() {
    let a = vec![1,2,3,4,];
    let b = vec![1,2,3,4,];

    dbg!(a.add(&b));
    dbg!(a.sub(&b));
    dbg!(a.transpose());

    let a = vec![1; 1_000_000];
    let b = vec![vec![10, 10]; 1_000_000];
    for _ in 0..100 {
        a.vec_mtx_mul(&b);
    }

}
