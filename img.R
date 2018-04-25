rm(list = ls())
setwd("D:\\OneDrive\\study\\data mining II\\project")

library(png)
library(imager)
files_m <- list.files(path="data//Pictures", pattern=".png",all.files=T, full.names=T, no.. = T)
images.male <- lapply(files_m, readPNG)
names_m <- gsub(".png","",gsub("data//Pictures/","",files_m))

males <- c()
for (i in 1:length(images.male)){
  m <- images.male[[i]][,,1]
  m <- as.vector(m)
  males <- rbind(males,m)
}
males.df <- as.data.frame(males)
males.df$sex <- 0

files_f <- list.files(path="data//Pictures_f", pattern=".png",all.files=T, full.names=T, no.. = T)
images.female <- lapply(files_f, readPNG)
names_f <- gsub(".png","",gsub("data//Pictures_f/","",files_f))

females <- c()
for (i in 1:length(images.female)){
  m <- images.female[[i]][,,1]
  m <- as.vector(m)
  females <- rbind(females,m)
}
females.df <- as.data.frame(females)
females.df$sex <- 1

q <- rbind(males.df,females.df)

save(file = "faces.RData",q)


####################
# PCA
####################
faces.pca <- prcomp(q[,2:ncol(q)],tol = 0.01)
dim(faces.pca$x)

dim(faces.pca$rotation)#[1] 16384   571

########################
# find best K for Kmeans
########################

set.seed(233)
# Compute and plot wss for k = 2 to k = 10.
k.max <- 10
wss <- sapply(1:k.max, 
              function(k){kmeans(faces.pca$x, k, nstart=50,iter.max = 1000 )$tot.withinss})
wss
X11()
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#when k = 4
kmean4 <- kmeans(faces.pca$x,4)
library(factoextra)
X11()
fviz_cluster(kmean4, data = faces.pca$x)

X11()
par(mfrow=c(2,2))
rot <- faces.pca$rotation
for(i in 1:nrow(kmean4$centers)){
  face.average <- kmean4$centers[i,]
  origin <- face.average %*% t(rot)
  a<-matrix(as.vector(origin),nrow = 128,ncol = 128)
  a<- apply(a, 2, rev)
  image(t(a),col = grey(seq(0, 1, length = 128)))
}


#when k = 5
kmean5 <- kmeans(faces.pca$x,5)
X11()
fviz_cluster(kmean5, data = faces.pca$x)

X11()
par(mfrow=c(3,2))
rot <- faces.pca$rotation
for(i in 1:nrow(kmean5$centers)){
  face.average <- kmean5$centers[i,]
  origin <- face.average %*% t(rot)
  a<-matrix(as.vector(origin),nrow = 128,ncol = 128)
  a<- apply(a, 2, rev)
  image(t(a),col = grey(seq(0, 1, length = 128)))
}

names.all <- c(names_m,names_f)
faces.all <- faces.pca$x
faces.all <- as.data.frame(faces.all,row.names = NULL)
faces.all$name <- names.all
faces.all$sex <- q$sex
write.table(faces.all, "faces_pca.csv", sep=",",row.names = F)#573col
write.table(faces.pca$rotation, "faces_pca_rotation.csv", sep=",",row.names = F)


####################
# faces vs weight
####################
full.data <- read.csv(file = "data/FullData.csv", header = TRUE)

exist.name <- subset(names.all, names.all %in% full.data$Name)


order.name <- match(exist.name,full.data$Name)
order.name <- order.name[!is.na(order.name)]

faces.all$Height <- rep(0,nrow(faces.all))
faces.all$Weight <- rep(0,nrow(faces.all))

for(i in 1:nrow(faces.all)){
  match.all <- full.data[full.data$Name == faces.all$name[i],]
  if(nrow(match.all)!=0 && !any(is.na(match.all))){
    faces.all$Height[i]<-match.all$Height
    faces.all$Weight[i]<-match.all$Weight
    print(match.all$Weight[i])
  }
}

faces.all <- subset(faces.all,faces.all$Height!=0)


library(glmnet)

train <- floor(0.6 * nrow(faces.all))
set.seed(122)
train_ind <- sample(seq_len(nrow(faces.all)), size = train)
train.set <- faces.all[train_ind,]
test.set <- faces.all[-train_ind,]

train.set.dup <- train.set
train.set.dup$name<- NULL
train.set.dup$sex<-NULL
train.set.dup$Height <- NULL

test.set.dup <- test.set
test.set.dup$name<- NULL
test.set.dup$sex<-NULL
test.set.dup$Height <- NULL

x<- as.matrix(train.set.dup[,1:ncol(train.set.dup)-1])
y<- as.matrix(train.set.dup[,ncol(train.set.dup)])

y.test <- as.matrix(test.set.dup[,ncol(test.set.dup)])
x.test <- as.matrix(test.set.dup[,1:ncol(test.set.dup)-1])
#Weight
fit <- glmnet(x,y,alpha=0, lambda=5,family = "gaussian")

error.train <- mean((y - predict(fit,x))^2)#57.36784 for lasso,0.5737272 for ridge
error.test <- mean((y.test - predict(fit,x.test))^2)#60.88833 for lasso, 60.24227 for ridge

train.set.dup <- train.set
train.set.dup$name<- NULL
train.set.dup$sex<-NULL
train.set.dup$Weight <- NULL

test.set.dup <- test.set
test.set.dup$name<- NULL
test.set.dup$sex<-NULL
test.set.dup$Weight <- NULL

x<- as.matrix(train.set.dup[,1:ncol(train.set.dup)-1])
y<- as.matrix(train.set.dup[,ncol(train.set.dup)])

y.test <- as.matrix(test.set.dup[,ncol(test.set.dup)])
x.test <- as.matrix(test.set.dup[,1:ncol(test.set.dup)-1])
#Height
fit <- glmnet(x,y,alpha=1, lambda=5,family = "gaussian")

error.train <- mean((y - predict(fit,x))^2)#48.626 for lasso,0.5814355 for ridge
error.test <- mean((y.test - predict(fit,x.test))^2)#50.04786 for lasso, 49.00541 for ridge
