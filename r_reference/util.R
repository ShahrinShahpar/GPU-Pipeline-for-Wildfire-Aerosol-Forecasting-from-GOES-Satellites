#-----------------------------------------------------------
# AVERAGE VELOCITY
#-----------------------------------------------------------
velocity <- function(dat.velocity, N.s, N.e){
  N_r <- sqrt(nrow(dat.velocity[[1]]))
  speed <- matrix(0, N_r^2, (N.e-N.s+1))
  s <- expand.grid(
    x = seq(0, 1-1/N_r, length.out = N_r),
    y = seq(0, 1-1/N_r, length.out = N_r)
  )
  for (i in N.s:N.e){
    speed[ ,(i-N.s+1)] <- dat.velocity[[i]]$speed
  }
  angle <- matrix(0, N_r^2, (N.e-N.s+1))
  for (i in N.s:N.e){
    angle[ ,(i-N.s+1)] <- dat.velocity[[i]]$angle
  }
  speed.avg <- matrix(apply(speed, 1, mean), N_r, N_r)
  angle.avg <- matrix(apply(angle, 1, mean), N_r, N_r)
  speed.avg.smooth <- image.smooth(speed.avg)$z/N_r
  angle.avg.smooth <- image.smooth(angle.avg)$z
  dat.velocity.avg.smooth <- data.frame(s, speed=c(speed.avg.smooth), angle=c(angle.avg.smooth))
  return(dat.velocity.avg.smooth)
}

# ----------------------------------------------------------
# GENERATE THE SET OF WAVENUMBERS
# ----------------------------------------------------------
Function_Omega <- function(N){
  Omega1 <- data.frame(
    k1 = c(0, 0, N/2, N/2),
    k2 = c(0, N/2, 0, N/2)
  )
  Omega2_p1 <- data.frame(
    k1 = seq(1, N/2-1, 1),
    k2 = N/2
  )
  Omega2_p2 <- expand.grid(
    k1 = seq(0, N/2, 1),
    k2 = seq(1, N/2-1, 1)
  ) 
  Omega2_p3 <- expand.grid(
    k1 = seq(1, N/2-1, 1),
    k2 = seq(-N/2+1, 0, 1)
  ) 
  Omega2 <- rbind(Omega2_p1, Omega2_p2, Omega2_p3)
  return(list(Omega1=Omega1, Omega2=Omega2))
}

# ----------------------------------------------------------
# CALCULATION OF THE FOURIER BASES
# ----------------------------------------------------------
Function_F <- function(N_rcr, N, Omega){
  Omega1 <- Omega$Omega1
  Omega2 <- Omega$Omega2
  sites_rc <- expand.grid(
    seq(0, 1-1/N_rcr, length.out = N_rcr),
    seq(0, 1-1/N_rcr, length.out = N_rcr)
  )
  F1 <- matrix(0, nrow(sites_rc), nrow(Omega1))
  for (i in 1:nrow(sites_rc)){
    for(j in 1:nrow(Omega1)){
      F1[i,j] <- cos(2*pi*sites_rc[i,1]*Omega1[j,1]+2*pi*sites_rc[i,2]*Omega1[j,2])
    }
  }
  F2c <- matrix(0, nrow(sites_rc), nrow(Omega2))
  for (i in 1:nrow(sites_rc)){
    for(j in 1:nrow(Omega2)){
      F2c[i,j] <- 2*cos(2*pi*sites_rc[i,1]*Omega2[j,1]+2*pi*sites_rc[i,2]*Omega2[j,2])
    }
  }
  F2s <- matrix(0, nrow(sites_rc), nrow(Omega2))
  for (i in 1:nrow(sites_rc)){
    for(j in 1:nrow(Omega2)){
      F2s[i,j] <- 2*sin(2*pi*sites_rc[i,1]*Omega2[j,1]+2*pi*sites_rc[i,2]*Omega2[j,2])
    }
  }
  rv <- cbind(F1, F2c, F2s)
  return(rv)
}

# ----------------------------------------------------------
# CALCULATION FOR THE FOURIER BASES
# ----------------------------------------------------------
FT <- function(dat, k1, k2){
  f_real <- function(n) {dat[n[1], n[2]]*cos(2*pi/N*(n[1]-1)*k1+2*pi/N*(n[2]-1)*k2)}
  f_imaginary <- function(n) {dat[n[1], n[2]]*sin(2*pi/N*(n[1]-1)*k1+2*pi/N*(n[2]-1)*k2)}
  N <- nrow(dat)
  n1 <- seq(1, N, 1)
  n2 <- seq(1, N, 1)
  z <- as.matrix(expand.grid(n1, n2))
  rv_real <- sum(apply(z, 1, f_real))/N^2
  rv_imaginary <- sum(apply(z, 1, f_imaginary))/N^2
  return(list(real = rv_real, imaginary = rv_imaginary))
}
Function_coef <- function(dat, N, Omega){
  Omega1 <- Omega$Omega1
  Omega2 <- Omega$Omega2
  coef1 <- numeric(nrow(Omega1))
  for(i in 1:nrow(Omega1)){
    coef1[i] <- FT(dat,Omega1[i,1],Omega1[i,2])$real
  }
  coef2c <- numeric(nrow(Omega2))
  for(i in 1:nrow(Omega2)){
    coef2c[i] <- FT(dat,Omega2[i,1],Omega2[i,2])$real
  }
  coef2s <- numeric(nrow(Omega2))
  for(i in 1:nrow(Omega2)){
    coef2s[i] <- FT(dat,Omega2[i,1],Omega2[i,2])$imaginary
  }
  rv <- c(coef1, coef2c, coef2s)
  return(rv)
}

# ----------------------------------------------------------
# TRANSITION MATRIX G CALCULATION WITH PHYSICAL PARAMETERS
# ----------------------------------------------------------
Phi1.int <- function(k1, k2, m1, m2, delta, v) {
  f <- function(s) {2*pi*(s[3]*k1+s[4]*k2)*sin(2*pi*s[1]*k1+2*pi*s[2]*k2)*cos(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row)
  z <- as.matrix(cbind(expand.grid(x, y), v))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
Phi2.int <- function(k1, k2, m1, m2, delta, v) {
  f <- function(s) {-2*pi*(s[3]*k1+s[4]*k2)*cos(2*pi*s[1]*k1+2*pi*s[2]*k2)*cos(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row) 
  z <- as.matrix(cbind(expand.grid(x, y), v))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
Phi3.int <- function(k1, k2, m1, m2, delta, v) {
  f <- function(s) {2*pi*(s[3]*k1+s[4]*k2)*sin(2*pi*s[1]*k1+2*pi*s[2]*k2)*sin(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row)
  z <- as.matrix(cbind(expand.grid(x, y), v))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
Phi5.int <- function(k1, k2, m1, m2, delta, K.ifm) {
  f <- function(s) {(-4*pi^2*(k1^2+k2^2)*s[3]*cos(2*pi*s[1]*k1+2*pi*s[2]*k2)-2*pi*(s[4]*k1+s[5]*k2)*sin(2*pi*s[1]*k1+2*pi*s[2]*k2))*cos(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row)
  z <- as.matrix(cbind(expand.grid(x, y), K.ifm))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
Phi6.int <- function(k1, k2, m1, m2, delta, K.ifm) {
  f <- function(s) {(-4*pi^2*(k1^2+k2^2)*s[3]*sin(2*pi*s[1]*k1+2*pi*s[2]*k2)-2*pi*(s[4]*k1+s[5]*k2)*cos(2*pi*s[1]*k1+2*pi*s[2]*k2))*cos(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row)
  z <- as.matrix(cbind(expand.grid(x, y), K.ifm))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
Phi7.int <- function(k1, k2, m1, m2, delta, K.ifm) {
  f <- function(s) {(-4*pi^2*(k1^2+k2^2)*s[3]*cos(2*pi*s[1]*k1+2*pi*s[2]*k2)-2*pi*(s[4]*k1+s[5]*k2)*sin(2*pi*s[1]*k1+2*pi*s[2]*k2))*sin(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row)
  z <- as.matrix(cbind(expand.grid(x, y), K.ifm))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
Phi8.int <- function(k1, k2, m1, m2, delta, K.ifm) {
  f <- function(s) {(-4*pi^2*(k1^2+k2^2)*s[3]*sin(2*pi*s[1]*k1+2*pi*s[2]*k2)-2*pi*(s[4]*k1+s[5]*k2)*cos(2*pi*s[1]*k1+2*pi*s[2]*k2))*sin(2*pi*s[1]*m1+2*pi*s[2]*m2)}
  N.row <- 1/delta
  x <- seq(0, 1-1/N.row, length.out = N.row)
  y <- seq(0, 1-1/N.row, length.out = N.row)
  z <- as.matrix(cbind(expand.grid(x, y), K.ifm))
  rv <- sum(apply(z, 1, f))*delta^2
  return(rv)
}
G_ad <- function(delta, v, K.ifm, Omega){ 
  Omega1 <- Omega$Omega1
  Omega2 <- Omega$Omega2
  N_Gr <- 2*nrow(Omega2) + nrow(Omega1)
  G <- matrix(0, N_Gr, N_Gr)
  k1 <- nrow(Omega1)
  k2 <- nrow(Omega2)
  # for the real part in Omega1 
  # for the special case, m = 1
  for (k in 1:k1){
    G[1, k] <- Phi1.int(Omega1[k,1],Omega1[k,2],Omega1[1,1],Omega1[1,2],delta,v) +
      Phi5.int(Omega1[k,1],Omega1[k,2],Omega1[1,1],Omega1[1,2],delta,K.ifm)
  }
  for (k in (k1+1):(k1+k2)){
    G[1, k] <- 2*Phi1.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega1[1,1],Omega1[1,2],delta,v) + 
      2*Phi5.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega1[1,1],Omega1[1,2],delta,K.ifm)
  }
  for (k in (k1+k2+1):(k1+2*k2)){
    G[1, k] <- 2*Phi2.int(Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],Omega1[1,1], Omega1[1,2],delta,v) + 
      2*Phi6.int(Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],Omega1[1,1], Omega1[1,2],delta,K.ifm)
  }
  # for m!=1 in Omega1 
  for (m in 2:k1){
    for (k in 1:k1){
      G[m, k] <- 2*Phi1.int(Omega1[k,1],Omega1[k,2],Omega1[m,1],Omega1[m,2],delta,v) + 
        2*Phi5.int(Omega1[k,1],Omega1[k,2],Omega1[m,1],Omega1[m,2],delta,K.ifm)
    }
    for (k in (k1+1):(k1+k2)){
      G[m, k] <- 4*Phi1.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega1[m,1],Omega1[m,2],delta,v) + 
        4*Phi5.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega1[m,1],Omega1[m,2],delta,K.ifm)
    }
    for (k in (k1+k2+1):(k1+2*k2)){
      G[m, k] <- 4*Phi2.int(Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],Omega1[m,1],Omega1[m,2],delta,v) + 
        4*Phi6.int(Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],Omega1[m,1],Omega1[m,2],delta,K.ifm)
    }
  }
  # for the real part in Omega2 
  for (m in 1:k2){
    for (k in 1:k1){
      G[m+k1, k] <- Phi1.int(Omega1[k,1],Omega1[k,2],Omega2[m,1],Omega2[m,2],delta,v) + 
        Phi5.int(Omega1[k,1],Omega1[k,2],Omega2[m,1],Omega2[m,2],delta,K.ifm)
    }
    for (k in (k1+1):(k1+k2)){
      G[m+k1, k] <- 2*Phi1.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega2[m,1],Omega2[m,2],delta,v) + 
        2*Phi5.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega2[m,1],Omega2[m,2],delta,K.ifm)
    }
    for (k in (k1+k2+1):(k1+2*k2)){
      G[m+k1, k] <- 2*Phi2.int(Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],Omega2[m,1],Omega2[m,2],delta,v) +
        2*Phi6.int(Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],Omega2[m,1],Omega2[m,2],delta,K.ifm)
    }
  }
  # for the imaginary part in Omega2
  for (m in 1:k2){
    for (k in 1:k1){
      G[m+k1+k2, k] <- Phi3.int(Omega1[k,1],Omega1[k,2],Omega2[m,1],Omega2[m,2],delta,v) + 
        Phi7.int(Omega1[k,1],Omega1[k,2],Omega2[m,1],Omega2[m,2],delta,K.ifm)
    }
    for (k in (k1+1):(k1+k2)){
      G[m+k1+k2, k] <- 2*Phi3.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega2[m,1],Omega2[m,2],delta,v) + 
        2*Phi7.int(Omega2[k-k1,1],Omega2[k-k1,2],Omega2[m,1],Omega2[m,2],delta,K.ifm)
    }
    for (k in (k1+k2+1):(k1+2*k2)){
      G[m+k1+k2, k] <- -2*Phi1.int(Omega2[m,1],Omega2[m,2],Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],delta,v) +
        2*Phi8.int(Omega2[m,1],Omega2[m,2],Omega2[k-(k1+k2),1],Omega2[k-(k1+k2),2],delta,K.ifm)
    }
  }
  return(G)
}
G_nit <- function(Omega){
  Omega1 <- Omega$Omega1
  Omega2 <- Omega$Omega2
  N_Gr <- 2*nrow(Omega2) + nrow(Omega1)
  G <- matrix(0, N_Gr, N_Gr)
  k1 <- nrow(Omega1)
  k2 <- nrow(Omega2)
  #for real part in Omega2 
  for (m in 1:k2){
    for (k in (k1+k2+1):(k1+2*k2)){
      if (k-(k1+k2) == m){
        G[m+k1, k] <- -2*pi*(Omega2[k-(k1+k2),1]+Omega2[k-(k1+k2),2])
      }
      else{
        G[m+k1, k] <- 0
      }
    }
  }
  ##for imaginary part in Omega2
  for (m in 1:k2){
    for (k in (k1+1):(k1+k2)){
      if (k-k1 == m){
        G[m+k1+k2, k] <- 2*pi*(Omega2[(k-k1),1]+Omega2[(k-k1),2])
      }
      else{
        G[m+k1+k2, k] <- 0
      }
    }
  }
  return(G)
}

# ----------------------------------------------------------
# THE SIGNAL FUNCTION FOR SIMULATION
# ----------------------------------------------------------
Source <- function(size, range, x_src1, x_src2, x_src3, x_src4, x) {
  s1 = size
  h1 = range
  s2 = size
  h2 = range
  s3 = size
  h3 = range
  s4 = size
  h4 = range
  output <- numeric(nrow(x))
  for(i in 1:nrow(x)) {
    output[i] = s1/(2*pi*h1^2)*exp(-norm(x_src1-x[i, ],type="2")/(2*h1^2)) + 
      ifelse(is.na(x_src2),0,s2/(2*pi*h2^2)*exp(-norm(x_src2-x[i, ],type = "2")/(2*h2^2)))[1] +
      ifelse(is.na(x_src3),0,s3/(2*pi*h3^2)*exp(-norm(x_src3-x[i, ],type = "2")/(2*h3^2)))[1] +
      ifelse(is.na(x_src4),0,s4/(2*pi*h4^2)*exp(-norm(x_src4-x[i, ],type = "2")/(2*h4^2)))[1]
  }
  return(output)
}

# ----------------------------------------------------------
# DATA PRE-PROCESSING FOR GIBBS-FFBS
# ----------------------------------------------------------
UDS <- function(datset, n.slt){
  test <- datset[[1]]
  long.range <- range(test$long)
  lat.range <- range(test$lat)
  lat.slt = seq(lat.range[1],lat.range[2], 0.04)[60/n.slt*(1:n.slt)]
  long.slt = seq(long.range[1],long.range[2], 0.04)[60/n.slt*(1:n.slt)]
  for (i in 1:length(datset)){
    datset[[i]]$id <- 1:nrow(datset[[i]])
    s.slt <- datset[[i]] %>%
      subset(lat%in%round(lat.slt,digits=2) & long%in%round(long.slt,digits=2))
    datset[[i]]$AOD <- NaN
    datset[[i]]$AOD[s.slt$id] <- s.slt$AOD
  }
  return(datset)
}
M_id <- function(y){                    
  ms.id <- which(is.na(y)*1==0)
  M <- matrix(0, length(ms.id), length(y))
  for (i in 1:length(ms.id)){
    M[i, ms.id[i]] <- 1
  }
  return(list(M=M, ms.id=ms.id))
}
OBS_ccl1 <- function(y, F){
  T <- length(y)
  Ft <- list()
  y.c <- list()
  id <- list()
  for (i in 1:T){
    tempt <- M_id(y[[i]])
    Ft[[i]] <- tempt$M%*%F
    y.c[[i]] <- y[[i]][tempt$ms.id]
    id[[i]] <- tempt$ms.id
  }
  return(list(Ft=Ft,y.c=y.c,id=id))
}
OBS_ccl2 <- function(y1, y2, F){
  T <- length(y1)
  F1t <- list()
  F2t <- list()
  y1.c <- list() 
  y2.c <- list()
  id1 <- list()
  id2 <- list()
  for (i in 1:T){
    tempt1 <- M_id(y1[[i]])
    tempt2 <- M_id(y2[[i]])
    F1t[[i]] <- tempt1$M%*%F
    F2t[[i]] <- tempt2$M%*%F
    y1.c[[i]] <- y1[[i]][tempt1$ms.id]
    y2.c[[i]] <- y2[[i]][tempt2$ms.id]
    id1[[i]] <- tempt1$ms.id
    id2[[i]] <- tempt2$ms.id
  }
  return(list(F1t=F1t,F2t=F2t,y1.c=y1.c,y2.c=y2.c,id1=id1,id2=id2))
}

# ----------------------------------------------------------
# GIBBS SAMPLING WITH FFBS 
# ----------------------------------------------------------
Kalman_Filter <- function(y, F, G, V, W, m0, C0){
  T <- ncol(y)
  m.flt = m.prd = C.flt = C.prd = vector("list", length = T+1)
  m.flt[[1]] <- m0
  C.flt[[1]] <- C0
  for (t in 1:T){
    m.prd[[t]] <- G%*%m.flt[[t]]
    C.prd[[t]] <- G%*%C.flt[[t]]%*%t(G) + W
    Q <-  F%*%C.prd[[t]]%*%t(F) + V
    Q.inv <- solve(Q)
    m.flt[[t+1]] <- m.prd[[t]]+C.prd[[t]]%*%t(F)%*%Q.inv%*%(y[,t]-F%*%m.prd[[t]])
    C.flt[[t+1]] <- C.prd[[t]]-C.prd[[t]]%*%t(F)%*%Q.inv%*%F%*%C.prd[[t]]
  }
  m.prd[[T+1]] <- G%*%m.flt[[t]]
  C.prd[[T+1]] <- G%*%C.flt[[t]]%*%t(G) + W
  return(list(m.flt=m.flt, m.prd=m.prd, C.flt=C.flt, C.prd=C.prd))
}
Gibbs_FFBS_DR1 = function(y, F, m0, C0, N.sample, N){
  T <- ncol(y)
  sigma2_v <- numeric(N.sample)
  W <- replicate(N.sample, matrix(0, N^2, N^2))
  G <- replicate(N.sample, matrix(0, N^2, N^2))  # create a storage for MCMC samples
  alpha = 8
  beta = 1
  nu <- N^2
  Phi <- diag(0.01, N^2)  
  sigma2_v[1] <- 0.01
  W[,,1] <- riwish(nu, Phi)
  G[,,1] <- riwish(nu, Phi) # initialization
  
  for (k in 2:N.sample) {
    V <- diag(sigma2_v[k-1], nrow(y))
    fit.kf <- Kalman_Filter(y, F, G[,,k-1], V, W[,,k-1], m0, C0)  # forward filtering 
    
    m.flt <- fit.kf$m.flt 
    m.prd <- fit.kf$m.prd
    C.flt <- fit.kf$C.flt 
    C.prd <- fit.kf$C.prd
    theta <- matrix(0, N^2, T)
    theta[, T] <- t(mvrnorm(1, m.flt[[T+1]], C.flt[[T+1]]))
    for (t in (T-1):1){
      C.pred.iv <- solve(C.prd[[t+1]])
      ht <- m.flt[[t+1]]+C.flt[[t+1]]%*%t(G[,,k-1])%*%C.pred.iv%*%(theta[,t+1]-m.prd[[t+1]])
      Ht <- C.flt[[t+1]]-C.flt[[t+1]]%*%t(G[,,k-1])%*%C.pred.iv%*%G[,,k-1]%*%C.flt[[t+1]]
      theta[, t] <- t(mvrnorm(1, ht, Ht))
    }  # backward sampling
    
    beta = beta + sum((y-F%*%theta)^2)/2
    alpha = alpha + T*nrow(y)/2
    sigma2_v[k] <- rinvgamma(1, alpha, beta)  # update sigma2v
    
    Theta <- theta[,1:(T-1)]
    Theta_p <- theta[,2:T]
    r = 8
    Theta.svd <- svd(Theta, nu = r, nv = r)
    Sigma = diag(Theta.svd$d[1:r])
    U = Theta.svd$u
    V = Theta.svd$v
    w <- t(mvrnorm(T-1, rep(0, N^2), W[,,k-1]))
    G[,,k] = (Theta_p-w)%*%V%*%solve(Sigma)%*%t(U)  # calculate G 
    
    nu <- nu+T-1
    Phi <- Phi + (Theta_p-G[,,k]%*%Theta)%*%t(Theta_p-G[,,k]%*%Theta)
    W[,,k] <- riwish(nu, Phi)  # update W
  }
  G.temp <- diag(0,N^2)
  N_start <- round(1/4*N.sample+1)
  for (i in N_start:N.sample){
    G.temp <- G[,,i]+G.temp
  }
  G.fit <- G.temp/(length(N_start:N.sample))
  return(list(G=G.fit, m.flt=m.flt))
}
Gibbs_FFBS1 <- function(obs.ccl, G, m0, C0, N.sample){
  # input data 
  y <- obs.ccl$y
  T <- length(y)
  Ft <- obs.ccl$Ft
  id <- obs.ccl$id
  G <- rbind( 
    cbind(G, diag(nrow(G))),
    cbind(matrix(0,nrow(G),nrow(G)), diag(nrow(G)))
  )
  # initialization and storage for Gibbs sampler 
  sigma2_v <- numeric(N.sample)
  W <- list()
  alpha =  1
  beta = 0.1
  nu <- nrow(C0)
  Phi <- diag(0.01, nrow(C0))  
  sigma2_v[1] <- rinvgamma(1, alpha, beta)
  W[[1]] <- rwish(nu, Phi)
  # loop for Gibbs sampling
  for (k in 2:N.sample){
    # Forward Filtering
    m.flt = m.prd = C.flt = C.prd = vector("list")
    m.flt[[1]] <- m0
    C.flt[[1]] <- C0
    for (t in 1:T){
      m.prd[[t]] <- G%*%m.flt[[t]]
      C.prd[[t]] <- G%*%C.flt[[t]]%*%t(G) + W[[k-1]]
      Vt <- diag(sigma2_v[[k-1]], length(y[[t]]))
      Ftt <- cbind(Ft[[t]], matrix(0,nrow(Ft[[t]]),ncol(Ft[[t]])))
      Q <-  Ftt%*%C.prd[[t]]%*%t(Ftt) + Vt 
      Q.inv <- solve(Q)
      m.flt[[t+1]] <- m.prd[[t]]+C.prd[[t]]%*%t(Ftt)%*%Q.inv%*%(y[[t]]-Ftt%*%m.prd[[t]])
      C.flt[[t+1]] <- C.prd[[t]]-C.prd[[t]]%*%t(Ftt)%*%Q.inv%*%Ftt%*%C.prd[[t]]
    }
    m.prd[[T+1]] <- G%*%m.flt[[t]]
    C.prd[[T+1]] <- G%*%C.flt[[t]]%*%t(G) + W[[k-1]]
    # Backward Sampling
    theta <- matrix(0, nrow(G), T)
    theta[, T] <- t(mvrnorm(1, m.flt[[T+1]], C.flt[[T+1]]))
    for (t in (T-1):1){
      ht <- m.flt[[t+1]]+C.flt[[t+1]]%*%t(G)%*%solve(C.prd[[t+1]])%*%(theta[,t+1]-m.prd[[t+1]])
      Ht <- C.flt[[t+1]]-C.flt[[t+1]]%*%t(G)%*%solve(C.prd[[t+1]])%*%G%*%C.flt[[t+1]]
      theta[, t] <- t(mvrnorm(1, ht, Ht))
    }
    # residual calculation
    y.rsd.clss <- rep(0,T)
    y.rsd.cln <- rep(0,T)
    for (t in 1:T){
      tempt <- y[[t]]-Ft[[t]]%*%theta[1:(1/2*length(m0)),t]
      y.rsd.cln[t] <- length(tempt)
      y.rsd.clss[t] <- sum(tempt^2)
    }
    state.rsd <- matrix(0, length(m0), T-1)
    for (t in 1:(T-1)){
      state.rsd[,t] <- theta[,t+1] - G%*%theta[,t]
    }
    # update parameters
    alpha <- alpha + sum(y.rsd.cln)/2
    beta <- beta + sum(y.rsd.clss)/2
    sigma2_v[k] <- rinvgamma(1, alpha, beta)
    nu <- nu + T-1
    Phi <- Phi + state.rsd%*%t(state.rsd)
    W[[k]] <- riwish(nu, Phi)
  }
  return(list(W=W, sigma2_v= sigma2_v, m.flt=m.flt, G.A=G))
}
Gibbs_FFBS2 <- function(obs.ccl, G, m0, C0, N.sample){
  # input data 
  y1 <- obs.ccl$y1.c
  y2 <- obs.ccl$y2.c
  T <- length(y1)
  F1t <- obs.ccl$F1t
  F2t <- obs.ccl$F2t
  id1 <- obs.ccl$id1
  id2 <- obs.ccl$id2
  G <- rbind( 
    cbind(G, diag(nrow(G))),
    cbind(matrix(0,nrow(G),nrow(G)), diag(nrow(G)))
  )
  # initialization and storage for Gibbs sampler 
  sigma12_v <- numeric(N.sample)
  sigma22_v <- numeric(N.sample)
  W <- list()
  alpha1 =  alpha2 = 1
  beta1 = beta2 = 0.1
  nu <- nrow(C0)
  Phi <- diag(0.01, nrow(C0))  
  sigma12_v[1] <- rinvgamma(1, alpha1, beta1)
  sigma22_v[1] <- rinvgamma(1, alpha2, beta2)
  W[[1]] <- rwish(nu, Phi)
  # loop for Gibbs sampling
  for (k in 2:N.sample){
    # Forward Filtering
    m.flt = m.prd = C.flt = C.prd = vector("list")
    m.flt[[1]] <- m0
    C.flt[[1]] <- C0
    for (t in 1:T){
      m.prd[[t]] <- G%*%m.flt[[t]]
      C.prd[[t]] <- G%*%C.flt[[t]]%*%t(G) + W[[k-1]]
      sigma2t_v <- c(rep(sigma12_v[k-1], length(y1[[t]])), rep(sigma22_v[k-1], length(y2[[t]])))
      Vt <- diag(sigma2t_v)
      Ft <- rbind(F1t[[t]], F2t[[t]])
      Ftt <- cbind(Ft, matrix(0,nrow(Ft),ncol(Ft)))
      Q <-  Ftt%*%C.prd[[t]]%*%t(Ftt) + Vt 
      Q.inv <- solve(Q)
      yt <- c(y1[[t]], y2[[t]])
      m.flt[[t+1]] <- m.prd[[t]]+C.prd[[t]]%*%t(Ftt)%*%Q.inv%*%(yt-Ftt%*%m.prd[[t]])
      C.flt[[t+1]] <- C.prd[[t]]-C.prd[[t]]%*%t(Ftt)%*%Q.inv%*%Ftt%*%C.prd[[t]]
    }
    m.prd[[T+1]] <- G%*%m.flt[[t]]
    C.prd[[T+1]] <- G%*%C.flt[[t]]%*%t(G) + W[[k-1]]
    # Backward Sampling
    theta <- matrix(0, nrow(G), T)
    theta[, T] <- t(mvrnorm(1, m.flt[[T+1]], C.flt[[T+1]]))
    for (t in (T-1):1){
      ht <- m.flt[[t+1]]+C.flt[[t+1]]%*%t(G)%*%solve(C.prd[[t+1]])%*%(theta[,t+1]-m.prd[[t+1]])
      Ht <- C.flt[[t+1]]-C.flt[[t+1]]%*%t(G)%*%solve(C.prd[[t+1]])%*%G%*%C.flt[[t+1]]
      theta[, t] <- t(mvrnorm(1, ht, Ht))
    }
    # residual calculation
    y1.rsd.clss <- rep(0,T)
    y1.rsd.cln <- rep(0,T)
    for (t in 1:T){
      tempt <- y1[[t]]-F1t[[t]]%*%theta[1:(1/2*length(m0)),t]
      y1.rsd.cln[t] <- length(tempt)
      y1.rsd.clss[t] <- sum(tempt^2)
    }
    y2.rsd.clss <- rep(0,T)
    y2.rsd.cln <- rep(0,T)
    for (t in 1:T){
      tempt <- y2[[t]]-F2t[[t]]%*%theta[1:(1/2*length(m0)),t]
      y2.rsd.cln[t] <- length(tempt)
      y2.rsd.clss[t] <- sum(tempt^2)
    }
    state.rsd <- matrix(0, length(m0), T-1)
    for (t in 1:(T-1)){
      state.rsd[,t] <- theta[,t+1] - G%*%theta[,t]
    }
    # update parameters
    alpha1 <- alpha1 + sum(y1.rsd.cln)/2
    beta1 <- beta1 + sum(y1.rsd.clss)/2
    alpha2 <- alpha2 + sum(y2.rsd.cln)/2
    beta2 <- beta2 + sum(y2.rsd.clss)/2
    nu <- nu + T-1
    Phi <- Phi + state.rsd%*%t(state.rsd)
    sigma12_v[k] <- rinvgamma(1, alpha1, beta1)
    sigma22_v[k] <- rinvgamma(1, alpha2, beta2)
    W[[k]] <- riwish(nu, Phi)
  }
  return(list(W=W, sigma12_v=sigma12_v, sigma22_v=sigma22_v, m.flt=m.flt))
}

# ----------------------------------------------------------
# COMPARISON WITH SSDF APPROACH
# ----------------------------------------------------------
MSEs_DF_COMPARISON <- function(F, coef, G, bias){
  sigma_xi = 0.1
  N_step <- 29
  step <- 0.015
  G_step <- expm(step*G)
  alpha <- coef
  data <- F%*%alpha
  y_sim <- pmax(data + rnorm(nrow(F), 0, sigma_xi),0) 
  for (i in 1:N_step){
    alpha <- G_step%*%alpha
    tempt <- pmax(F%*%alpha + rnorm(nrow(F), 0, sigma_xi),0)
    y_sim <- cbind(y_sim, tempt)
  }
  
  if (sum(bias)==0){
    bias = 0
    noise_setting = list(c(0.36, 0.36), c(0.36, 3.6), c(3.6, 3.6))
  }else{
    noise_setting = list(c(0.36, 0.36), c(0.36, 3.6), c(3.6, 0.36), c(3.6, 3.6))
  }
  N_case = length(noise_setting)
  MSE_SSDF = matrix(0, N_case, 30)
  MSE_our = matrix(0, N_case, 30)
  for (j in 1:N_case){
    sigma_epsilon1 = noise_setting[[j]][1]
    sigma_epsilon2 = noise_setting[[j]][2]
    y1 = y_sim
    for (i in 1:30){
      y1[, i] = y_sim[, i] + rnorm(nrow(F), 0, sigma_epsilon1) + bias
    }
    y2 = y_sim
    for (i in 1:30){
      y2[, i] = y_sim[, i] + rnorm(nrow(F), 0, sigma_epsilon2)
    } 
    for (t in 1:30){
      rho = (sigma_epsilon1 / sigma_epsilon2)^2
      tempt = 1/(rho+1)*y1[, t] + rho/(rho+1)*y2[, t] - y_sim[, t]
      MSE_SSDF[j,t] = mean(tempt^2)
    }  
    y_1 <- list()
    y_2 <- list()
    for (i in 1:20){
      y_1[[i]] <- y1[, i]
      y_2[[i]] <- y2[, i]
    }
    obs.ccl <- OBS_ccl2(y_1, y_2, F)
    fit.M2 <- Gibbs_FFBS2(obs.ccl, expm(step*G), rep(0.1, 2*N^2), diag(0.01, 2*N^2), 10)
    F.A <- cbind(F, matrix(0, nrow(F), ncol(F)))
    for (t in 1:20){
      tempt = F.A%*%fit.M2$m.flt[[t+1]] - y_sim[, t]
      MSE_our[j,t] = mean(tempt^2)
    }
    dummy = F
    for (t in 21:30){
      G_temp = expm((t-20)*0.015*G)
      pred_error <- F%*%G_temp%*%fit.M2$m.flt[[21]][1:36] - y_sim[, t]
      MSE_our[j,t] = mean(pred_error^2)
    }
  }
  
  MSEs_DF_COMPARISON1 = matrix(0, N_case, 10)
  MSEs_DF_COMPARISON2 = matrix(0, N_case, 10)
  MSEs_DF_COMPARISON1[, c(1, 3, 5, 7, 9)] = round(MSE_SSDF[, 16:20], digits = 3)
  MSEs_DF_COMPARISON1[, c(2, 4, 6, 8, 10)] = round(MSE_our[, 16:20], digits = 3)
  MSEs_DF_COMPARISON2[, c(1, 3, 5, 7, 9)] = round(MSE_SSDF[, 21:25], digits = 3)
  MSEs_DF_COMPARISON2[, c(2, 4, 6, 8, 10)] = round(MSE_our[, 21:25], digits = 3)
  return(rbind(MSEs_DF_COMPARISON1, MSEs_DF_COMPARISON2))
}
