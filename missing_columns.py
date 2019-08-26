
# Options
# Data dictionary is missing columns
# posted in WiDS 2018 Datathon a day ago
# I noticed that the data dictionary is missing a couple hundred columns. There are 1069 column descriptions in the data dictionary, but there are 1235 columns in the data. There are also some columns that are listed in the data dictionary which do not appear in the training data.
#
# It says on the data description page that some columns were removed from the dictionary for privacy/leakage reasons, so it sounds like this is intentional.
#
# Most of the categorical columns in training data are encoded as integers, so I wonder if some of the columns that are missing descriptions in the data dictionary are also categorical but currently encoded as integer in the data. It would be great to know if any of the undefined integer columns are actually categorical (so I can re-encode them as factors/categorical before doing my modeling).
#
# Maybe a Kaggle admin can comment on this?
#
# Here's the full list of the differences below (in R).

library(readxl)
library(tidyverse)

df &lt;- read_csv("train.csv")
dd &lt;- read_excel("WiDS_data_dictionary.xlsx")

# Columns in data dictionary but not in training data
 &gt; setdiff(dd$`Column Name`, names(df))
 [1] "AA8"     "AA19"    "AA20"    "AA21"    "AB3"     "AB6"     "DL4_24"  "DL4_25"  "G2P2_17" "RI4"     "RI6_1"
[12] "RI6_2"   "RI6_3"   "RI7_1"   "RI7_2"   "RI7_3"   "RI8_1"   "RI8_2"   "RI8_3"   "RI8_4"   "RI8_5"   "RI8_6"
[23] "RI8_7"   "RI8_8"   "RI8_96"  "RI8_10"

# Columns in training data but not in data dictionary
&gt; setdiff(names(df), dd$`Column Name`)
   "train_id"

"AA4",             "AA7"              "AA14"             "AA15"             "DG1"
      "DG8a"             "DG8b"             "DG8c"             "DG9a"             "DG9b"             "DG9c"
 "DG10b"      ,      "DG10c"            "DG11b"            "DG11c"
  "DL4_96"           "DL4_99"                "DL8"              "DL11"
 ,    "DL14"                 "G2P2_96"

  "G2P3_1"
 "G2P3_2"         ,  "G2P3_3"           "G2P3_4"           "G2P3_5"           "G2P3_6"           "G2P3_7"           "G2P3_8"
"G2P3_9"           ,"G2P3_10"          "G2P3_11"          "G2P3_12"          "G2P3_13"          "G2P3_14"          "G2P3_15"
 "G2P3_16"          ,"G2P3_96"          "MT1"              "MT3_1"            "MT3_2"            "MT3_3"
     "MT6C"           "MT12_1"
 "MT12_2"           ,"MT12_3"           "MT12_4"           "MT12_5"           "MT12_6"           "MT12_7"           "MT12_8"
 "MT12_9"           ,"MT12_10"          "MT12_11"          "MT12_12"          "MT12_13"          "MT12_14"          "MT12_96"
 "MT12_99"
     "FF7_1"            "FF7_2"            "FF7_3"
 "FF7_4"           , "FF7_5"            "FF7_6"            "FF7_7"            "FF7_96"              "FF8_1"
 "FF8_2"           , "FF8_3"            "FF8_4"            "FF8_5"            "FF8_6"            "FF8_7"            "FF8_96"
 "MM12_OTHERS"      "MM12_REC"             "MM13_REC"      "MM23"
   "LN2_RIndLngBEOth" "LN2_WIndLngBEOth"