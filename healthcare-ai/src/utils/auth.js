// utils/auth.js
export const checkLoginStatus = () => {
    const token = localStorage.getItem("token");
    return token !== null;
  };
  