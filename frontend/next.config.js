// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // This is crucial for static HTML export
  images: {
    unoptimized: true, // Recommended for static export if using next/image
  },
  // IMPORTANT: For <username>.github.io, basePath and assetPrefix should NOT be set
  // or should be empty strings.
  // basePath: '',
  // assetPrefix: '',
};

module.exports = nextConfig;