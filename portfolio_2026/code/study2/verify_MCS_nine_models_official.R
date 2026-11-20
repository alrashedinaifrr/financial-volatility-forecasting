#!/usr/bin/env Rscript
# Official R MCS verification for the full nine-model 2026 forecast panel.
# Completed official Google Colab run supplied 18 Nov 2026; the resulting
# sessionInfo, 24 serialized fits, and all output tables are archived separately.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L)stop("Usage: Rscript verify_MCS_nine_models_official.R INPUT.csv OUTPUT_DIR")
input <- args[[1L]]
out <- args[[2L]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(!requireNamespace("MCS",quietly=TRUE))stop("MCS not installed. install.packages('MCS')")
raw <- read.csv(input,check.names=FALSE,stringsAsFactors=FALSE)
cols <- c("market","date","model","target_sq_return","pred_var")
stopifnot(all(cols %in% names(raw)))
stopifnot(!anyDuplicated(raw[,c("market","date","model")]))
stopifnot(all(is.finite(raw$target_sq_return)),all(raw$target_sq_return>=0))
stopifnot(all(is.finite(raw$pred_var)),all(raw$pred_var>0))
models <- c("Direct mean","EWMA","GARCH","HAR-SR2","Log target","Poisson","RF","RV20","Rescaled log")
stopifnot(setequal(unique(raw$model),models))
stopifnot(identical(sort(unique(raw$market)),c("SP500","TASI")))
sets <- list(all_nine=models,exclude_direct_mean=setdiff(models,"Direct mean"))
rows <- list()
model_rows <- list()
loss_check <- list()
counter <- 0L
for(market in c("SP500","TASI")) {
  sub <- raw[raw$market==market,,drop=FALSE]
  dates <- sort(unique(sub$date))
  expected_n <- if(market=="SP500") 195L else 188L
  stopifnot(length(dates)==expected_n)
  loss <- matrix(NA_real_,nrow=length(dates),ncol=length(models),dimnames=list(dates,models))
  for(i in seq_along(dates)) {
    part <- sub[sub$date==dates[[i]],,drop=FALSE]
    stopifnot(nrow(part)==length(models),setequal(part$model,models))
    stopifnot(max(part$target_sq_return)-min(part$target_sq_return)<1e-14)
    y <- part$target_sq_return[[1L]]
    for(j in seq_along(models)) {
      pred <- part$pred_var[part$model==models[[j]]]
      stopifnot(length(pred)==1L,is.finite(pred),pred>0)
      f <- max(pred,1e-9)
      loss[i,j] <- log(f)+y/f
    }
  }
  stopifnot(all(is.finite(loss)))
  write.csv(data.frame(date=dates,loss,check.names=FALSE),file.path(out,paste0(market,"_nine_model_QLIKE.csv")),row.names=FALSE)
  for(j in seq_along(models)) {
    loss_check[[length(loss_check)+1L]] <- data.frame(market=market,model=models[[j]],n=nrow(loss),mean_qlike=mean(loss[,j]))
  }
  for(set_name in names(sets)) for(stat in c("Tmax","TR")) for(k in c(5L,10L,20L)) {
    counter <- counter+1L
    case <- paste(market,set_name,stat,paste0("k",k),sep="_")
    set.seed(20261118L+counter)
    started <- proc.time()[["elapsed"]]
    fit <- tryCatch(MCS::MCSprocedure(Loss=loss[,sets[[set_name]],drop=FALSE],
       alpha=.05,B=4999L,statistic=stat,k=k,verbose=FALSE,
       seed=20261118L+counter),error=function(e)e)
    elapsed <- proc.time()[["elapsed"]]-started
    if(inherits(fit,"error")) {
      msg <- conditionMessage(fit)
      writeLines(msg,file.path(out,paste0(case,".error.txt")))
      rows[[counter]] <- data.frame(id=case,market=market,candidate_set=set_name,statistic=stat,k=k,
                  B=4999L,status="ERROR",included="",excluded="",error=msg,elapsed_seconds=elapsed)
      next
    }
    info <- methods::slot(fit,"Info")
    sh <- as.matrix(methods::slot(fit,"show"))
    # The official MCS 0.2.1 show slot contains Avg.Loss,
    # p-Value for H_{0,M_k}, and MCS p-Value (in that order).
    # Check column labels instead of silently misidentifying an index.
    expected_columns <- c("Avg.Loss", "p-Value for H_{0,M_k}", "MCS p-Value")
    if (!identical(colnames(sh), expected_columns)) {
       stop(paste("Unexpected official-MCS show-slot columns:",
                  paste(colnames(sh), collapse=" | ")))
    }
    keep <- as.character(info$included)
    drop <- as.character(info$excluded)
    stopifnot(setequal(c(keep,drop),sets[[set_name]]))
    saveRDS(fit,file.path(out,paste0(case,".rds")))
    writeLines(c(paste("CASE:",case),capture.output(print(fit)),
            paste("MCS package:",as.character(utils::packageVersion("MCS"))),
            paste("Included:",paste(keep,collapse="; ")),
            paste("Excluded:",paste(drop,collapse="; "))),file.path(out,paste0(case,".txt")))
    rows[[counter]] <- data.frame(id=case,market=market,candidate_set=set_name,statistic=stat,k=k,
            B=4999L,status="OK",included=paste(keep,collapse="; "),
            excluded=paste(drop,collapse="; "),error="",elapsed_seconds=elapsed)
    stopifnot(nrow(sh)==length(sets[[set_name]]))
    model_rows[[length(model_rows)+1L]] <- data.frame(id=case,market=market,
            candidate_set=set_name,statistic=stat,k=k,model=rownames(sh),
            mean_QLIKE=as.numeric(sh[,"Avg.Loss"]),
            raw_p=as.numeric(sh[,"p-Value for H_{0,M_k}"]),
            mcs_p=as.numeric(sh[,"MCS p-Value"]),
            included=rownames(sh) %in% keep)
  }
}
statuses <- do.call(rbind,rows)
write.csv(statuses,file.path(out,"run_status.csv"),row.names=FALSE)
write.csv(do.call(rbind,loss_check),file.path(out,"model_mean_qlike.csv"),row.names=FALSE)
if(length(model_rows))write.csv(do.call(rbind,model_rows),file.path(out,"all_model_pvalues.csv"),row.names=FALSE)
writeLines(c(capture.output(sessionInfo()),paste("MCS version:",as.character(utils::packageVersion("MCS")))),
           file.path(out,"sessionInfo.txt"))
cat("Official MCS version:",as.character(utils::packageVersion("MCS")),"\n")
cat("Finished:",nrow(statuses),"cases; errors:",sum(statuses$status!="OK"),"\n")
if(nrow(statuses)!=24L||any(statuses$status!="OK"))quit(status=2L)
