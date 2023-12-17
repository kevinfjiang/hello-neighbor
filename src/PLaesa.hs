{-# LANGUAGE ScopedTypeVariables, NamedFieldPuns, RecordWildCards #-}
module PLaesa (pInitLaesa, pPredict, Laesa) where

import Laesa (Laesa(..))
import MetricSpace (MetricSpace(..))

import Data.Ord (comparing)
import Data.List (maximumBy, minimumBy, transpose, length)
import Data.Heap (MinPrioHeap, fromList, view)
import Control.Parallel.Strategies (using, rdeepseq)

strat = rdeepseq  -- constant strat to be chosen and changed later

keyMax :: Ord k => [k] -> [m] -> (k, m)
keyMax key el = maximumBy (comparing fst) (zip key el)

keyMin :: Ord k => [k] -> [m] -> (k, m)
keyMin key el = minimumBy (comparing fst) (zip key el)

pInitLaesa :: MetricSpace m -> Int -> Laesa m
pInitLaesa ms@MetricSpace{mData, mDist} numBases =
  Laesa{
    lMetricSpace = ms, lNumBases = numBases,
    lBases = fst <$> baseIndicesAndDists,
    lBaseDists = snd <$> baseIndicesAndDists
  }
  where rawBaseIndicesAndDists = (head mData, [], replicate (length mData) 0) :
          [ (maxDistBase, currBaseDists, lowerBounds) |
            (currBase, _, prevLowerBounds) <- rawBaseIndicesAndDists,

            let currBaseDists = map (mDist currBase) mData `using` strat,
            let lowerBounds = zipWith (+) currBaseDists prevLowerBounds `using` strat,
            let maxDistBase = snd $ keyMax lowerBounds mData]
        baseIndicesAndDists = take numBases $ map (\(e1, e2, _) -> (e1, e2)) rawBaseIndicesAndDists

computeLowerBounds :: [[Double]] -> [Double] -> [Double]
computeLowerBounds baseDist targDist = map maxLB (transpose baseDist) `using` strat
  where maxLB = (maximum).(\x -> zipWith (+) targDist x `using` strat)

pPredict :: forall m. Laesa m -> m -> m
pPredict Laesa{lMetricSpace=MetricSpace{..}, ..} target = bestFromBound (view minHeap) (keyMin targDist lBases)
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

