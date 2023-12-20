{-# LANGUAGE ScopedTypeVariables, NamedFieldPuns, RecordWildCards #-}
module PLaesa (pInitLaesa, pPredict) where

import Laesa (Laesa(..))
import MetricSpace (MetricSpace(..))

import Data.List (maximumBy, minimumBy, transpose)
import Data.Ord (comparing)

import Control.Parallel.Strategies (using, rdeepseq, rparWith, parBuffer)
import Data.Heap (MinPrioHeap, fromList, view)
import Data.IntSet (IntSet, insert, notMember, singleton)

keyMax :: Ord k => [(k, m)] -> (k, m)
keyMax = maximumBy (comparing fst)

keyMin :: Ord k => [(k, m)] -> (k, m)
keyMin = minimumBy (comparing fst)


pInitLaesa :: MetricSpace m -> Int -> Laesa m
pInitLaesa ms@MetricSpace{mData} numBases = Laesa{
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
        newLowerBound = zipWith (+) lowerBounds currBaseDists `using` parBuffer 16 rdeepseq
        (maxIndex, newBase) = snd $ keyMax $ filter (\(_, (index, _)) -> notMember index visited) $
          zip newLowerBound $ zip [0..] mData

computeLowerBounds :: [[Float]] -> [Float] -> [Float]
computeLowerBounds baseDist targDist = map maxLB baseDistT `using` parBuffer 16 rdeepseq
  where maxLB = maximum.zipWith (\a b -> abs a-b) targDist
        baseDistT = transpose baseDist

pPredict :: forall m. Laesa m -> m -> m
pPredict Laesa{lMetricSpace=MetricSpace{..}, ..} target =
  bestFromBound (view minHeap) (keyMin $ zip targDist lBases)
  where lowerBounds = computeLowerBounds lBaseDists targDist
        targDist = [mDist target base | base <- lBases] `using` parBuffer 16 rdeepseq
        minHeap = fromList $ zip lowerBounds mData

        bestFromBound :: Maybe ((Float, m), MinPrioHeap Float m) -> (Float, m) -> m
        bestFromBound Nothing (_, bestCand) = bestCand
        bestFromBound (Just ((currLb, curr), remain)) (bestDist, best)
          | currLb > bestDist = best
          | currDist < bestDist = bestFromBound (view remain) (currDist, curr)
          | otherwise = bestFromBound (view remain) (bestDist, best)
          where currDist = mDist curr target

