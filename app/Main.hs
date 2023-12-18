module Main (main) where

import Lib

main :: IO()
main = do
  searchBase <- readVecs getFVec "data/siftsmall_base.fvecs"
  queries <- readVecs getFVec "data/siftsmall_query.fvecs"
  target <- return $ queries !! 3
  top <- return $ head searchBase

  ms <- return $ euclideanSpace $ take 9999 searchBase
  model <- return $ pInitLaesa ms 1000
  pred1 <- return $ pPredict model top
  pred2 <- return $ pPredict model target

  print "In base search"
  print top
  print pred1
  print "Out base search"
  print target
  print pred2

  return ()


