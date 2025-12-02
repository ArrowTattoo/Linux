library(palmerpenguins)
data("penguins")
#clean
penguins_clean <- na.omit(penguins)
penguins_quant <- penguins_clean[, c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")]

penguins_scaled <- scale(penguins_quant)
set.seed(123)

kmeans_result <- kmeans(penguins_scaled, centers = 3, nstart = 25)
print(kmeans_result)

penguins_clean$cluster <- as.factor(kmeans_result$cluster)

plot(penguins_scaled[, "bill_length_mm"], 
     penguins_scaled[, "bill_depth_mm"], 
     col = kmeans_result$cluster, 
     pch = 19, 
     main = "K-means Simple Plot", 
     xlab = "Bill Length", 
     ylab = "Bill Depth")

# pch = 8 : 星号形状
# cex = 2 : 放大 2 倍
points(kmeans_result$centers[, c(1,2)], 
       col = "black", 
       pch = 8, 
       cex = 2)

conf_matrix <- table(Predicted_Cluster = penguins_clean$cluster, 
                     Actual_Species = penguins_clean$species)
print(conf_matrix)
plot(penguins_quant, col = kmeans_result$cluster, pch = 19)