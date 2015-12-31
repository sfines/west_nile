#
#@author Steven Fines
#@version 1.0
#@descr Standard tools and utilities to configure a new R workspace
#with the right packages.
#
r_major <- as.numeric(R.version$major)
r_minor <- as.numeric(R.version$minor)

lappend <- function(lst, obj) {
  lst[[length(lst)+1]] <- obj
  return(lst)
}

required.packages <- c(
  'reshape2', #useful for reshaping data also includes plyr
  'ggplot2', #best. Graphing. Ever.
  'ggmap', #Adds geographical mapping to ggplot
  'digest',  #Adds cryptographical verification
  'knitr', #Document generation for notebook-style work
  'dplyr', #Specialized tools for data manipulation
  'tidyr', #pipelines for data formatting
  'ggvis', #Grammar of graphics dynamic visualization
  'markdown', #add markdown language support to knitr
  'iterators', 
  'foreach', 
  'zoo', 
  'maps', 
  'geosphere',
  'ROCR',
  'grid',
  'randomForest',
  'cvTools'
  )

packages.to.install <- NULL

for(i in 1:length(required.packages)){
  pkg <- required.packages[i]
  if(library(pkg, logical.return=T, character.only=T)){
    #Unload Package
    detach(pos=match(paste('package', pkg, sep=":"), search()))
  } else {
    packages.to.install <- lappend(packages.to.install, pkg)    
  }
}


if(length(packages.to.install) > 0){
  print(paste('Installing ', length(pacakges.to.install), ' packages.'))
  install.packages(packages.to.install)
} else {
  print('Not installing any packages')
}

remove(required.packages)
remove(packages.to.install)
remove(pkg)
remove(i)

if(r_major >= 3 && r_minor >= 1.0){
  if(!library(devtools, logical.return=T)){
    install.packages("devtools")
  }
  library(devtools)
  if( !library(magrittr, logical.return=T)){
    install_github("smbache/magrittr")
  }
  
  detach(package:devtools)
} else { 
  print(paste(R.Version()$version.string, 
              "does not have an installable devtools, upgrade R"))
}

remove(r_major)
remove(r_minor)

