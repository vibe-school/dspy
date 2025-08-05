# DSPy Tutorial Site Deployment Guide

## Build Status ✅
The site has been successfully built and is ready for deployment.

## Build Output
- Location: `guide/dspy-tutorial-site/build/`
- The build folder contains all static files needed for deployment

## Deployment Options

### 1. GitHub Pages (Recommended)
```bash
# From the dspy-tutorial-site directory
npm run deploy
```

### 2. Netlify
1. Drag and drop the `build` folder to [Netlify](https://app.netlify.com/drop)
2. Or use the Netlify CLI:
```bash
netlify deploy --dir=build --prod
```

### 3. Vercel
```bash
vercel --prod build
```

### 4. Traditional Web Server
Simply upload the contents of the `build` folder to your web server's public directory.

## Local Testing
To test the production build locally:
```bash
cd guide/dspy-tutorial-site
npm run serve
```
Then open http://localhost:3000/dspy-den/

## Important URLs
- Base URL: `/dspy-den/`
- Homepage: `https://your-domain.com/dspy-den/`
- Tutorial Start: `https://your-domain.com/dspy-den/tutorial/intro`

## Environment Variables
No environment variables are required for the static site deployment.

## Post-Deployment Checklist
- [ ] Verify all pages load correctly
- [ ] Check navigation links work
- [ ] Test on mobile devices
- [ ] Confirm images load properly
- [ ] Check 404 page works