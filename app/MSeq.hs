module Main (main) where

import Lib
import System.Environment(getEnv)
import Text.Read(readMaybe)

main :: IO()
main = do
  searchBase <- readVecs getFVec "data/siftsmall_base.fvecs"
  queries <- readVecs getFVec "data/siftsmall_query.fvecs"

  [rawNumTrain, rawNumBase, rawNumQuery] <- mapM getEnv ["NUM_TRAIN", "NUM_BASES", "NUM_QUERY"]
  (numTrain, numBase, numQuery) <- return $
    case map readMaybe [rawNumTrain, rawNumBase, rawNumQuery] of
        [Just iNumTrain, Just iNumBase, Just iNumQuery] -> (iNumTrain, iNumBase, iNumQuery)  -- Love type inference
        _ -> error "Environment variables for `NUM_BASE` and `NUM_TRAIN` must be integers"


  let ms = euclideanSpace $ take numTrain searchBase
      model = initLaesa ms numBase

      searches = take numQuery queries
      predicts = map (predict model) searches

  print $ "Average loss: " ++ show (zipWith (mDist ms) searches predicts)

  return ()


