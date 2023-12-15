module Lib (buildPair) where

import Data.Array(Array, listArray, (!), (//))

buildPair :: (Int, Int)
buildPair = let arr  = listArray (1,10) (repeat 37) :: Array Int Int
                arr' = arr // [(1, 64)]
            in (arr ! 1, arr' ! 1)
