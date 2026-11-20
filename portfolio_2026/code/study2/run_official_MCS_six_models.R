#!/usr/bin/env Rscript
# Official-package validation of 2026 MCS for two equity markets.
# Base R + MCS only; outputs preserve both raw package objects and readable logs.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)<3) stop('Usage: Rscript run_official_MCS.R predictions_2026.csv INPUT_DIR OUTPUT_DIR')
input_path <- args[[1]]
input_dir <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
if (!requireNamespace('MCS',quietly=TRUE)) stop('CRAN package MCS not installed')
all_models <- c('Direct mean','EWMA','HAR-SR2','Log target','Poisson','Rescaled log')
raw <- read.csv(input_path,check.names=FALSE, stringsAsFactors=FALSE)
need <- c('market','date','model','target_sq_return','pred_var')
stopifnot(all(need %in% names(raw)))
stopifnot(all(is.finite(raw$target_sq_return)),all(raw$target_sq_return>=0))
stopifnot(all(is.finite(raw$pred_var)),all(raw$pred_var>0))
stopifnot(!anyDuplicated(raw[,c('market','date','model')]))
stopifnot(identical(sort(unique(raw$market)),c('SP500','TASI')))
stopifnot(setequal(unique(raw$model), all_models))

run_info <- list()
model_sets <- list(all_six=all_models,
                   exclude_direct_mean=setdiff(all_models,'Direct mean'),
                   four_core=c('EWMA','HAR-SR2','Poisson','Rescaled log'))
rows <- list()
loss_comparisons <- list()
case_index <- 0L
cat('MCS version:',as.character(packageVersion('MCS')),'\n')

for(market in c('SP500','TASI')) {
    sub <- raw[raw$market==market,,drop=FALSE]
    dates <- sort(unique(sub$date))
    exp_n <- if(market=='SP500') 195L else 188L
    stopifnot(length(dates)==exp_n)
    loss <- matrix(NA_real_,nrow=length(dates),ncol=length(all_models),
                   dimnames=list(dates,all_models))
    y <- numeric(length(dates))
    for(i in seq_along(dates)) {
        sl <- sub[sub$date==dates[[i]],,drop=FALSE]
        stopifnot(nrow(sl)==6L,setequal(sl$model,all_models))
        stopifnot(max(sl$target_sq_return)-min(sl$target_sq_return)<1e-14)
        y[[i]] <- sl$target_sq_return[[1L]]
        for(j in seq_along(all_models)) {
            z <- sl$pred_var[sl$model==all_models[[j]]]
            stopifnot(length(z)==1,is.finite(z),z>0)
            z <- max(z,1e-9)
            loss[i,j] <- log(z)+y[[i]]/z
        }
    }
    stopifnot(all(is.finite(loss)))
    write.csv(data.frame(date=dates,loss,check.names=FALSE),
              file.path(output_dir,paste0(market,'_qlike_from_R.csv')),
              row.names=FALSE,quote=TRUE)
    ref <- read.csv(file.path(input_dir,paste0('reference_',market,'_qlike.csv')),
                    check.names=FALSE,stringsAsFactors=FALSE)
    stopifnot(identical(as.character(ref$date), as.character(dates)),
              identical(colnames(ref)[-1], all_models))
    dif <- max(abs(unname(as.matrix(ref[,-1,drop=FALSE])) - unname(loss)))
    stopifnot(is.finite(dif),dif<1e-7)
    loss_comparisons[[market]]<-data.frame(market=market,n=length(dates),
                           max_abs_qlike_difference=dif,
                           mean_qlike_summary=paste(sprintf('%s=%.6f',
                           all_models,colMeans(loss)),collapse=' ; '))

    for(set_name in names(model_sets)) {
      columns <- model_sets[[set_name]]
      for(stat in c('Tmax','TR')) {
        for(k in c(5L,10L,20L)) {
          case_index <- case_index+1L
          case_id <- sprintf('%s_%s_%s_k%d',market,set_name,stat,k)
          cat('Running',case_id,'\n'); flush.console()
          set.seed(20261118L+case_index)
          logpath <- file.path(output_dir,paste0(case_id,'.txt'))
          fn <- function() MCS::MCSprocedure(Loss=loss[,columns,drop=FALSE],
                  alpha=0.05, B=4999L, statistic=stat, k=k, verbose=FALSE)
          started <- proc.time()[['elapsed']]
          obj <- tryCatch(fn(),error=function(e)e)
          elapsed <- proc.time()[['elapsed']]-started
          if(inherits(obj,'error')) {
             writeLines(conditionMessage(obj),logpath)
             rows[[case_index]]<-data.frame(id=case_id,market=market,candidate_set=set_name,
              stat=stat,k=k,B=4999L,nrow=nrow(loss),status='ERROR',error=conditionMessage(obj),
              elapsed_seconds=elapsed)
             next
          }
          saveRDS(obj,file.path(output_dir,paste0(case_id,'.rds')))
          log_lines <- capture.output({cat('CASE',case_id,'\n'); print(obj);
             cat('\nS4 slots: ',paste(slotNames(obj),collapse=', '),'\n');
             for(s in slotNames(obj)) {cat('\nSLOT: ',s,'\n'); print(slot(obj,s))}
          })
          writeLines(log_lines,logpath)
          rows[[case_index]]<-data.frame(id=case_id,market=market,candidate_set=set_name,
              stat=stat,k=k,B=4999L,nrow=nrow(loss),status='OK',error='',
              elapsed_seconds=elapsed)
        }
      }
    }
}
write.csv(do.call(rbind,rows),file.path(output_dir,'run_status.csv'),row.names=FALSE)
write.csv(do.call(rbind,loss_comparisons),file.path(output_dir,'loss_matrix_comparison.csv'),row.names=FALSE)
writeLines(c(capture.output(sessionInfo()),
              paste('MCS version:',as.character(packageVersion('MCS')))),
              file.path(output_dir,'sessionInfo.txt'))
fail <- sum(vapply(rows,function(x)x$status[[1]]!='OK',logical(1)))
cat('Finished. Cases:',length(rows),'Errors:',fail,'\n')
if(fail)quit(status=2L)
