import React from "react";

export default function Card({ children, className }) {
  return (
    <div className={`bg-greenCard rounded-[14px] shadow-lg ${className}`}>
      {children}
    </div>
  );
}
