module MetricSpace (DistMetric, MetricSpace(..)) where

type DistMetric m = m -> m -> Float

data MetricSpace m = MetricSpace{
  mData :: [m],  -- candidates
  mDist :: DistMetric m
}

euclidean :: [Float] -> [Float] -> Float
euclidean a b = sqrt $ euclideanSquare a b

euclideanSquare :: [Float] -> [Float] -> Float
euclideanSquare [] [] = 0
euclideanSquare (a: as) (b: bs) = (a-b)**2 + euclideanSquare as bs
euclideanSquare _ _ = error "Vectors must be of the same length"

euclideanSpace :: [[Float]] -> MetricSpace [Float]
euclideanSpace candidates = MetricSpace{mData=candidates, mDist=euclidean}