#langs=("de" "fi" "fr" "hr" "it" "ru" "tr" "en" "bg" "ca" "eo" "et" "eu" "he" "hu" "id" "ka" "ko" "lt" "no" "th")
langs=('af' 'az' 'et' 'eu' 'gu' 'ht' 'jv' 'ka' 'kk' 'ml' 'mr' 'my' 'pa' 'qu' 'sw' 'te' 'th' 'tl' 'wo' 'yo')
for f in ${langs[@]}
do
	echo Downloading $f
	wget -P /ceph/tgreen/projects/babelnet_extraction/vocab https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$f.vec
done
