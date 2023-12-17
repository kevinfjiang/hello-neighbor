module MetricSpace (DistMetric, MetricSpace(..)) where

type DistMetric m = m -> m -> Double

data MetricSpace m = MetricSpace{
  mData :: [m],  -- candidates
  mDataCount :: Int,
  mDist :: DistMetric m
}