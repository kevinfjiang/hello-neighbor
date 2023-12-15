module Laesa () where

import Data.Array (Array, listArray, (!), (//))
import Data.Foldable (toList)
import Data.Ord (comparing)
import Data.List (maximumBy)

type DistMetric m = m -> m -> Double

data MetricSpace m = MetricSpace{
  mData :: Array Int m,
  mDist :: DistMetric m
}

data Laesa m = Laesa{
  lMetricSpace :: MetricSpace m,
  lNumBases :: Int,
  lBaseIndices :: Array Int Int,
  lBaseDists :: Array (Int, Int) Double
}

initLaesa :: MetricSpace m -> Int -> Laesa m
initLaesa ms numBases =
  Laesa{
    lMetricSpace = ms,
    lNumBases = numBases,
    lBaseIndices = listArray (1, numBases) (map fst baseIndicesAndDists),
    lBaseDists = listArray ((1, 1), (numBases, msDataCount)) (concatMap snd baseIndicesAndDists)
  }
  where msData = mData ms
        msDataCount = length msData
        msDist = mDist ms
        msDataList = toList msData
        indexOfMax lst = fst $ maximumBy (comparing snd) (zip [0..] lst)
        rawBaseIndicesAndDists = (0, [], replicate msDataCount 0) : 
          [ (maxDistIndex, currBaseDists, lowerBounds) | 
            (baseIndex, _, prevLowerBounds) <- rawBaseIndicesAndDists,
            let currBase = msData ! baseIndex, 
            let currBaseDists = map (msDist currBase) msDataList, 
            let lowerBounds = zipWith (+) currBaseDists prevLowerBounds, 
            let maxDistIndex = indexOfMax lowerBounds ]
        baseIndicesAndDists = map (\(e1, e2, _) -> (e1, e2)) (tail rawBaseIndicesAndDists)

{--
foldUpdateSelect :: [m] -> DistMetric -> [[Double]] -> [Double]


-- computeBase :: MetricSpace m -> Int -> Int -> [Double] ->
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

--}