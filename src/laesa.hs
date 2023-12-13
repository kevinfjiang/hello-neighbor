-- TODO move this to metric space, separate
type DistMetric = m -> m -> Double

-- TODO make it something indexable
data MetricSpace m = MetricSpace{mData :: [m], mDist :: DistMetric}

data Laesa m = Laesa{
    lMetricSpace :: MetricSpace m,
    lNumBase :: Int,
    lNumData :: Int,
    lBase :: [Int],
    lBaseDist :: [Double],
}

foldUpdateSelect :: [m] -> DistMetric -> [[Double]] -> [Double]


computeBase :: MetricSpace m -> Int -> Int -> [Double] ->
computeBase ms baseRemain cand lowerBounds =
  | baseRemain == 0 =
  | otherwise =
  where
    ()
    candidates = mData ms
    foldUpdateSelect (oldMax, oldDist, oldBound) i =
      | i in visited =
      | otherwise = (max oldMax updatedBound, update i oldDist dist, update i oldBound updatedBound)
      where
        dist = (mDist ms) candidates[i] candidates[cand[-1]]
        updatedBound =  oldBound[i] + dist

buildLaesa :: MetricSpace m -> Int -> Laesa
buildLaesa ms numBase = Laesa{
  lMetricSpace=ms, lNumData=length (mData ms), lNumBase=numBase, lbase=base, lBaseDist=baseDict}
  where base, baseDict = computeBase ms numBase

nearestNeighbor :: Laesa m -> m -> m -- TODO custom untilM where we compare
nearestNeighbor laesa taret =
  where lowerBounds = computeLowerBounds