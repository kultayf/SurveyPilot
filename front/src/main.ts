import { createApp } from "vue";

import App from "./App.vue";
import router from "./router";
import "./styles.css";
import "./visual-refresh.css";

createApp(App).use(router).mount("#app");
