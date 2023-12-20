module MetricSpace (DistMetric, MetricSpace(..), euclideanSpace) where

type DistMetric m = m -> m -> Float

data MetricSpace m = MetricSpace{
  mData :: [m],  -- candidates
  mDist :: DistMetric m
}

euclideanDist :: [Float] -> [Float] -> Float
euclideanDist as bs = sqrt $ sum $ zipWith (\a b -> (a-b)**2) as bs

euclideanSpace :: [[Float]] -> MetricSpace [Float]
euclideanSpace candidates = MetricSpace{mData=candidates, mDist=euclideanDist}