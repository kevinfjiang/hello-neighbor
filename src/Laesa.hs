module Laesa (MetricSpace, initLaesa, predict) where

import Data.Array (listArray, (!))
import Data.Ord (comparing)
import Data.List (maximumBy, minimumBy)
import qualified Data.Heap as Heap

type DistMetric m = m -> m -> Double

data MetricSpace m = MetricSpace{
  mData :: [m],  -- candidates
  mDataCount :: Int,
  mDist :: DistMetric m
}

data Laesa m = Laesa{
  lMetricSpace :: MetricSpace m,
  lNumBases :: Int,
  lBaseIndices :: [Int],
  lBaseDists :: [[Double]]  -- numBases x numCandidates
}

initLaesa :: MetricSpace m -> Int -> Laesa m
initLaesa ms numBases =
  Laesa{
    lMetricSpace = ms,
    lNumBases = numBases,
    lBaseIndices = map fst baseIndicesAndDists,
    lBaseDists = map snd baseIndicesAndDists
  }
  where msData = mData ms
        msDataCount = mDataCount ms
        msDist = mDist ms
        msDataArr = listArray (1, msDataCount) msData
        indexOfMax lst = fst $ maximumBy (comparing snd) (zip [1..] lst)

        rawBaseIndicesAndDists = (1, [], replicate msDataCount 0) :
          [ (maxDistIndex, currBaseDists, lowerBounds) |
            (baseIndex, _, prevLowerBounds) <- rawBaseIndicesAndDists,
            let currBase = msDataArr ! baseIndex,
            let currBaseDists = map (msDist currBase) msData,
            let lowerBounds = zipWith (+) currBaseDists prevLowerBounds,
            let maxDistIndex = indexOfMax lowerBounds ]
        baseIndicesAndDists = map (\(e1, e2, _) -> (e1, e2)) (tail rawBaseIndicesAndDists)

predict :: Laesa m -> m -> m
predict la target = msDataArr ! i
  where laMetricSpace = lMetricSpace la
        -- laNumBases = lNumBases la
        laBaseIndices = lBaseIndices la
        laBaseDists = lBaseDists la
        msData = mData laMetricSpace
        msDataCount = mDataCount laMetricSpace
        msDist = mDist laMetricSpace
        msDataArr = listArray (1, msDataCount) msData
        -- lBaseDistsArr = listArray ((1, 1), (laNumBases, msDataCount)) (concat laBaseDists)
        absDiff a b = abs $ a - b
        indexAndMin lst = minimumBy (comparing snd) (zip [1..] lst)

        targetDists = map (\ind -> msDist target (msDataArr ! ind)) laBaseIndices
        -- targetDistsArr = listArray (1, laNumBases) targetDists
        bounds = zipWith (map . absDiff) targetDists laBaseDists  -- (map . absDiff) == (\td bds -> map (absDiff td) bds)
        lowerBounds = foldr (zipWith max) (replicate msDataCount 0) bounds
        lowerBoundsArr = listArray (1, msDataCount) lowerBounds
        candIndexWithBestDist lowerBoundHeap bestCandIndex bestDist
          | Heap.null lowerBoundHeap = bestCandIndex
          | otherwise =
              let candIndex = Heap.payload (Heap.minimum lowerBoundHeap)
              in if lowerBoundsArr ! candIndex > bestDist then bestCandIndex
              else let newLowerBoundHeap = Heap.deleteMin lowerBoundHeap
              in let newDist = msDist (msDataArr ! candIndex) target
              in if newDist < bestDist then candIndexWithBestDist newLowerBoundHeap candIndex newDist
              else candIndexWithBestDist newLowerBoundHeap bestCandIndex bestDist
        (baseIndex, baseDist) = indexAndMin targetDists
        baseCandIndex = laBaseIndices !! baseIndex
        i = candIndexWithBestDist (Heap.fromList (zipWith Heap.Entry lowerBounds [1..])) baseCandIndex baseDist
