source('./relief.R')

# minkowski_distance: Return Minkowski distance with parameter p between examples e1 and e2
minkowski_distance <- function(e1, e2, p) {
  return(sum(abs(e1 - e2)^p)^(1/p))
}

# Load test data
test_data <- read.table('./data/rba_test_data2.m')

# Perform feature ranking using Relief
weights <- relief(test_data[,-ncol(test_data)], test_data[,ncol(test_data)], 50, function(e1, e2) { minkowski_distance(e1, e2, 2) })