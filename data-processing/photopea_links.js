
// Script to generate link for opening PHotopea with images
// node index.js "lik1,link2"
function generateLink(links) {
  const array_links = links.split(",")
  console.log(array_links)
  const data = {
    files: array_links,
    script: "alert('Loaded document!')",
    "environment": {
      "topt": [0, 0, [16, true, true]]
    }
  }

  const data_encode = encodeURI(JSON.stringify(data));
  const url = 'https://www.photopea.com#' + data_encode;
  return url
}

var links = process.argv[2]
const array_links = links.split(",")
const photopea_link = `https://skytruth.surge.sh/?raster_file=${array_links[0]}&raster_overlap=${array_links[1]}`
console.log(photopea_link)
