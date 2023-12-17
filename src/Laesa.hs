{-# LANGUAGE ScopedTypeVariables, NamedFieldPuns, RecordWildCards #-}
module Laesa (Laesa(..), initLaesa, predict) where

import MetricSpace (MetricSpace(..))

import Data.Ord (comparing)
import Data.List (maximumBy, minimumBy, transpose)

import Data.Heap (MinPrioHeap, fromList, view)
import Data.IntSet (IntSet, insert, member, singleton)

data Laesa m = Laesa{
  lMetricSpace :: MetricSpace m,
  lNumBases :: Int,
  lBases :: [m],
  lBaseDists :: [[Double]]  -- numBases x numCandidates
}

keyMax :: Ord k => [(k, m)] -> (k, m)
keyMax = maximumBy (comparing fst)

keyMin :: Ord k => [(k, m)] -> (k, m)
keyMin = minimumBy (comparing fst)

initLaesa :: MetricSpace m -> Int -> Laesa m
initLaesa ms@MetricSpace{mData} numBases =
  Laesa{
    lMetricSpace = ms,
    lNumBases = numBases,
    lBases = fst <$> takeBases,
    lBaseDists = snd <$> takeBases
  }
  where initLowerBounds = replicate (length mData) 0
        basesFromSpace = initLaesaHelper ms (head mData) initLowerBounds (singleton 0)
        takeBases = take numBases basesFromSpace

initLaesaHelper :: MetricSpace m -> m -> [Double] -> IntSet -> [(m, [Double])]
initLaesaHelper ms@MetricSpace{..} currBase lowerBounds visited = (currBase, currBaseDists) :
  initLaesaHelper ms newBase newLowerBound (insert maxIndex visited)

  where currBaseDists = map (mDist currBase) mData
        newLowerBound = zipWith (+) lowerBounds currBaseDists
        (maxIndex, newBase) = snd $ keyMax $ filter (\(_, (index, _)) -> member index visited) $
          zip newLowerBound $ zip [0..] mData


computeLowerBounds :: [[Double]] -> [Double] -> [Double]  -- TODO add documentation
computeLowerBounds baseDist targDist = map maxLB (transpose baseDist)
  where maxLB = (maximum).(zipWith (+) targDist)

predict :: forall m. Laesa m -> m -> m
predict Laesa{lMetricSpace=MetricSpace{..}, ..} target =
  bestFromBound (view minHeap) (keyMin $ zip targDist lBases)
  where lowerBounds = computeLowerBounds lBaseDists targDist
        targDist = [mDist target base | base <- lBases]
        minHeap = fromList $ zip lowerBounds mData

        bestFromBound :: Maybe ((Double, m), MinPrioHeap Double m) -> (Double, m) -> m
        bestFromBound Nothing (_, bestCand) = bestCand
        bestFromBound (Just ((currLb, curr), remain)) (bestDist, best)
          | currLb > bestDist = best
          | currDist < bestDist = bestFromBound (view remain) (currDist, curr)
          | otherwise = bestFromBound (view remain) (bestDist, best)
          where currDist = mDist curr target

