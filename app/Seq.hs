module Main (main) where

import Lib
import System.Environment(getEnv)
import Text.Read(readMaybe)

main :: IO()
main = do
  searchBase <- readVecs getFVec "data/siftsmall_base.fvecs"
  queries <- readVecs getFVec "data/siftsmall_query.fvecs"

  rawNumTrain <- getEnv "NUM_TRAIN"
  rawNumBase <- getEnv "NUM_BASES"
  (numTrain, numBase) <- return $ case (readMaybe rawNumTrain, readMaybe rawNumBase) of
    (Just iNumTrain, Just iNumBase) -> (iNumTrain, iNumBase)  -- Love type inference
    _ -> error "Environment variables for `NUM_BASE` and `NUM_TRAIN` must be integers"


  let ms = euclideanSpace $ take numTrain searchBase
      model = initLaesa ms numBase

      search = head queries
      predicted = predict model search

  print $ "Target vector" ++ show search
  print $ "Predicted vector: " ++ show predicted

  return ()


