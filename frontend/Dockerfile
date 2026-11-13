# Stage 1: Build
FROM node:22-alpine AS builder

WORKDIR /app

# Copy dependency files first for layer caching
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies using the lockfile (exact versions)
RUN npm ci

# Copy source code
COPY frontend/ ./

# Build production bundle
ENV VITE_API_BASE=/api
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.27-alpine

# Copy production build to Nginx web root
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy Nginx configuration for SPA routing and API proxy
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Health check: Nginx serves index.html
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
