// 安装时立即激活
self.addEventListener("install", function(e) {
    self.skipWaiting();
});

// 激活时清空所有旧缓存
self.addEventListener("activate", function(e) {
    e.waitUntil(
        caches.keys().then(function(keys) {
            return Promise.all(keys.map(function(key) {
                return caches.delete(key);
            }));
        }).then(function() {
            return self.clients.claim();
        })
    );
});

// Network First 策略：优先网络，网络失败才用缓存
self.addEventListener("fetch", function(e) {
    e.respondWith(
        fetch(e.request)
            .then(function(response) {
                // 网络成功，直接返回
                return response;
            })
            .catch(function() {
                // 网络失败，尝试读缓存
                return caches.match(e.request);
            })
    );
});
