import React from "react";

export default function Loader({ size = 50 }) {
  return (
    <div
      className="spinner-container relative"
      style={{ height: size, width: size }}
    >
      <div className="mx-auto w-full h-full border-[7px] z-[1] border-    absolute rounded-[50%]"></div>
      <div className="spinner w-full h-full border-[7px] border-b-0 z-[2] border-greenText border-x-[transparent] absolute top-0 rounded-[50%]"></div>
    </div>
  );
}
