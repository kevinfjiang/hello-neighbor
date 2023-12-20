{-# LANGUAGE ScopedTypeVariables, NamedFieldPuns, RecordWildCards #-}
module Laesa (Laesa(..), initLaesa, predict) where

import MetricSpace (MetricSpace(..))

import Data.Ord (comparing)
import Data.List (maximumBy, minimumBy, transpose)

import Data.Heap (MinPrioHeap, fromList, view)
import Data.IntSet (IntSet, insert, notMember, singleton)

data Laesa m = Laesa{
  lMetricSpace :: MetricSpace m,
  lNumBases :: Int,
  lBases :: [m],
  lBaseDists :: [[Float]]  -- numBases x numCandidates
}

keyMax :: Ord k => [(k, m)] -> (k, m)
keyMax = maximumBy (comparing fst)

keyMin :: Ord k => [(k, m)] -> (k, m)
keyMin = minimumBy (comparing fst)

initLaesa :: MetricSpace m -> Int -> Laesa m
initLaesa ms@MetricSpace{mData} numBases = Laesa{
  lMetricSpace = ms,
  lNumBases = numBases,
  lBases = fst <$> takeBases,
  lBaseDists = snd <$> takeBases}
  where initLowerBounds = replicate (length mData) 0
        basesFromSpace = initLaesaHelper ms (head mData) initLowerBounds (singleton 0)
        takeBases = take numBases basesFromSpace

initLaesaHelper :: MetricSpace m -> m -> [Float] -> IntSet -> [(m, [Float])]
initLaesaHelper ms@MetricSpace{..} currBase lowerBounds visited = (currBase, currBaseDists) :
  initLaesaHelper ms newBase newLowerBound (insert maxIndex visited)

  where currBaseDists = map (mDist currBase) mData
        newLowerBound = zipWith (+) lowerBounds currBaseDists
        (maxIndex, newBase) = snd $ keyMax $ filter (\(_, (index, _)) -> notMember index visited) $
          zip newLowerBound $ zip [0..] mData


computeLowerBounds :: [[Float]] -> [Float] -> [Float]  -- TODO add documentation
computeLowerBounds baseDist targDist = map maxLB (transpose baseDist)
  where maxLB = maximum.zipWith (\a b -> abs a-b) targDist

predict :: forall m. Laesa m -> m -> m
predict Laesa{lMetricSpace=MetricSpace{..}, ..} target =
  bestFromBound (view minHeap) (keyMin $ zip targDist lBases)
  where lowerBounds = computeLowerBounds lBaseDists targDist
        targDist = [mDist target base | base <- lBases]
        minHeap = fromList $ zip lowerBounds mData

        bestFromBound :: Maybe ((Float, m), MinPrioHeap Float m) -> (Float, m) -> m
        bestFromBound Nothing (_, bestCand) = bestCand
        bestFromBound (Just ((currLb, curr), remain)) (bestDist, best)
          | currLb > bestDist = best
          | currDist < bestDist = bestFromBound (view remain) (currDist, curr)
          | otherwise = bestFromBound (view remain) (bestDist, best)
          where currDist = mDist curr target

